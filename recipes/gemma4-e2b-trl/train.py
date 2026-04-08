"""Fine-tune with plain TRL + PEFT (no DeepSpeed).

Simplest possible training setup — single GPU, standard PyTorch optimizer.
Good baseline for smaller models that fit in memory without ZeRO partitioning.

Launch via:
    python train.py [args...]
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
# Monkey-patch pynvml for DGX Spark GB10 (NVML not fully supported)
# ---------------------------------------------------------------------------
try:
    import pynvml
    _orig_nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo

    class _FakeMemInfo:
        def __init__(self):
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
            return _orig_nvmlDeviceGetMemoryInfo(handle, version) if version else _orig_nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError:
            return _FakeMemInfo()

    pynvml.nvmlDeviceGetMemoryInfo = _patched_nvmlDeviceGetMemoryInfo
except ImportError:
    pass
# ---------------------------------------------------------------------------

# Flush NFS page cache after safetensors loads
try:
    import safetensors.torch
    _orig_load_file = safetensors.torch.load_file
    def _load_and_flush(filename, *args, **kwargs):
        result = _orig_load_file(filename, *args, **kwargs)
        try:
            fd = os.open(str(filename), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)  # POSIX_FADV_DONTNEED
            finally:
                os.close(fd)
        except (OSError, AttributeError):
            pass
        return result
    safetensors.torch.load_file = _load_and_flush
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with TRL + PEFT (no DeepSpeed)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="/workspace/outputs")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
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
    return p.parse_known_args()[0]


def format_conversations(example, tokenizer, max_seq_length):
    """Format conversation data. Supports ShareGPT (from/value) and OpenAI (role/content)."""
    messages = []
    for t in example["conversations"]:
        if "from" in t:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            messages.append({"role": role_map.get(t["from"], t["from"]), "content": t["value"]})
        else:
            messages.append({"role": t.get("role", ""), "content": t["content"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors=None)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    gc.collect()

    # Replace custom linear layers (Gemma 4 compatibility)
    import torch.nn as nn
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear) and type(module).__name__ != 'Linear':
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], module.linear)
                replaced += 1
    if replaced:
        print(f"Replaced {replaced} custom linear wrappers with nn.Linear for LoRA")

    # LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Dataset
    raw = load_dataset("json", data_files=args.dataset, split="train")
    tokenized = raw.map(
        lambda ex: format_conversations(ex, tokenizer, args.max_seq_length),
        remove_columns=raw.column_names, num_proc=4, desc="Tokenizing",
    )

    eval_dataset = None
    if args.eval_fraction > 0:
        split = tokenized.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = tokenized

    # Train — no DeepSpeed, just plain PyTorch
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True, fp16=False,
            optim="adamw_torch",
            warmup_steps=5, logging_steps=1,
            save_strategy="steps", save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            weight_decay=0.01, lr_scheduler_type="linear",
            seed=args.seed, max_length=args.max_seq_length,
            packing=False, report_to="none",
            dataset_text_field=None,
            skip_memory_metrics=True,
        ),
    )

    print(f"Starting training: {len(train_dataset)} examples")
    trainer.train()

    trainer.save_model(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
    print(f"LoRA adapter saved to {args.output_dir}/lora_adapter")


if __name__ == "__main__":
    main()
