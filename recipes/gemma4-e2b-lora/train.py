"""Fine-tune with DeepSpeed ZeRO + LoRA.

Generic training script for DGX Manager fine-tune recipes.
Supports any HuggingFace causal LM model with LoRA adapters.

Launch via:
    /path/to/launch.sh [args...]
"""

import argparse
import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Monkey-patch safetensors to flush NFS page cache after each shard load.
# Without this, NFS page cache can accumulate and eat into GPU headroom.
# ---------------------------------------------------------------------------
POSIX_FADV_DONTNEED = 4


def _drop_page_cache(filepath):
    try:
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


try:
    import safetensors.torch
    _orig_load_file = safetensors.torch.load_file

    def _load_file_and_drop_cache(filename, *args, **kwargs):
        result = _orig_load_file(filename, *args, **kwargs)
        _drop_page_cache(filename)
        return result

    safetensors.torch.load_file = _load_file_and_drop_cache
except ImportError:
    pass
# ---------------------------------------------------------------------------


def flush_page_cache():
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except PermissionError:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with DeepSpeed ZeRO + LoRA")
    p.add_argument("--model_name", required=True, help="HF model ID or local path")
    p.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    p.add_argument("--output_dir", default="/workspace/outputs")
    p.add_argument("--ds_config", default=None, help="DeepSpeed config JSON path")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_fraction", type=float, default=0.1)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_known_args()[0]


def format_sharegpt(example, tokenizer, max_seq_length):
    """Format ShareGPT-style conversations for training."""
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    messages = [
        {"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
        for t in example["conversations"]
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(
        text, truncation=True, max_length=max_seq_length, return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    args = parse_args()
    world_rank = int(os.environ.get("RANK", 0))

    # ---- DeepSpeed config (optional) ----
    if args.ds_config:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        dschf = HfDeepSpeedConfig(args.ds_config)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    print(f"[Rank {world_rank}] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[Rank {world_rank}] Model loaded.")

    gc.collect()
    flush_page_cache()

    # ---- LoRA ----
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if world_rank == 0:
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ---- Dataset ----
    raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenized = raw_dataset.map(
        lambda ex: format_sharegpt(ex, tokenizer, args.max_seq_length),
        remove_columns=raw_dataset.column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    eval_dataset = None
    if args.eval_fraction > 0:
        split = tokenized.train_test_split(
            test_size=args.eval_fraction, seed=args.seed
        )
        train_dataset, eval_dataset = split["train"], split["test"]
        if world_rank == 0:
            print(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = tokenized

    # ---- Training ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        warmup_steps=5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
        report_to="none",
        dataset_text_field=None,
        deepspeed=args.ds_config,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    if world_rank == 0:
        print(f"Starting training: {len(train_dataset)} examples, "
              f"max_seq_length={args.max_seq_length}, batch_size={args.batch_size}")

    trainer.train()

    if world_rank == 0:
        trainer.save_model(f"{args.output_dir}/lora_adapter")
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter")


if __name__ == "__main__":
    main()
