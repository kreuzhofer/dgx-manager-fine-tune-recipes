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

# ---------------------------------------------------------------------------
# Monkey-patch pynvml for DGX Spark GB10 which doesn't support some NVML calls.
# DeepSpeed calls nvmlDeviceGetMemoryInfo during optimizer init which fails
# on the unified memory architecture. Patch to return total system RAM instead.
# ---------------------------------------------------------------------------
try:
    import pynvml
    _orig_nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo

    class _FakeMemInfo:
        def __init__(self):
            import shutil
            total, used, free = shutil.disk_usage("/")
            # Use system RAM as proxy
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            self.total = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            self.free = int(line.split()[1]) * 1024
                self.used = self.total - self.free
            except Exception:
                self.total = 128 * 1024**3
                self.free = 64 * 1024**3
                self.used = self.total - self.free

    def _patched_nvmlDeviceGetMemoryInfo(handle, version=None):
        try:
            if version:
                return _orig_nvmlDeviceGetMemoryInfo(handle, version)
            return _orig_nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError:
            return _FakeMemInfo()

    pynvml.nvmlDeviceGetMemoryInfo = _patched_nvmlDeviceGetMemoryInfo
except ImportError:
    pass
# ---------------------------------------------------------------------------

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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


class LogMetricsCallback(TrainerCallback):
    """Force-print training metrics to stdout for progress tracking."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            total = state.max_steps if state.max_steps > 0 else "?"
            loss = logs.get("loss", "?")
            lr = logs.get("learning_rate", "?")
            print(f"[TRAIN] step={step}/{total} loss={loss} lr={lr}", flush=True)
        if logs and "eval_loss" in logs:
            print(f"[EVAL] eval_loss={logs['eval_loss']}", flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        print("[EVAL] Running evaluation...", flush=True)


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


def format_example(example, tokenizer, max_seq_length):
    """Format a dataset example into tokenized chat for training.

    Supports multiple dataset formats:
    - ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
    - OpenAI:   {"conversations": [{"role": "user", "content": "..."}, ...]}
    - QA:       {"question": "...", "answer": "..."} with optional "context"
    - Instruct: {"instruction": "...", "output": "..."} with optional "input"
    """
    messages = []

    if "conversations" in example:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        for t in example["conversations"]:
            if "from" in t:
                messages.append({"role": role_map.get(t["from"], t["from"]), "content": t["value"]})
            else:
                messages.append({"role": role_map.get(t.get("role", ""), t.get("role", "")), "content": t["content"]})
    elif "question" in example and "answer" in example:
        user_msg = example["question"]
        if example.get("context"):
            user_msg = f"{example['context']}\n\n{example['question']}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["answer"]},
        ]
    elif "instruction" in example and "output" in example:
        user_msg = example["instruction"]
        if example.get("input"):
            user_msg = f"{example['instruction']}\n\n{example['input']}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]},
        ]
    else:
        raise ValueError(f"Unknown dataset format. Columns: {list(example.keys())}")

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
    # Set a default chat template if not present (e.g., Gemma 4)
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )

    # ---- Model ----
    print(f"[Rank {world_rank}] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[Rank {world_rank}] Model loaded.")

    gc.collect()
    flush_page_cache()

    # ---- Replace custom linear layers with standard nn.Linear for LoRA compatibility ----
    # Gemma 4 uses Gemma4ClippableLinear wrappers that PEFT doesn't support.
    # Unwrap them to standard nn.Linear before applying LoRA.
    import torch.nn as nn
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear) and type(module).__name__ != 'Linear':
            parent_name = name.rsplit('.', 1)
            if len(parent_name) == 2:
                parent = dict(model.named_modules())[parent_name[0]]
                setattr(parent, parent_name[1], module.linear)
                replaced += 1
    if replaced > 0 and world_rank == 0:
        print(f"Replaced {replaced} custom linear wrappers with nn.Linear for LoRA")

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
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        raw_dataset = load_dataset(args.dataset, split="train")

    if world_rank == 0:
        print(f"Dataset columns: {raw_dataset.column_names}, {len(raw_dataset)} examples")

    tokenized = raw_dataset.map(
        lambda ex: format_example(ex, tokenizer, args.max_seq_length),
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
        skip_memory_metrics=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[LogMetricsCallback()],
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
