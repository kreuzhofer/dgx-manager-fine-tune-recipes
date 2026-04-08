"""Fine-tune with Unsloth — fast LoRA/QLoRA training.

Unsloth provides 2x faster training and 60% less memory usage
compared to standard HuggingFace + PEFT.

Launch via:
    /path/to/launch.sh [args...]
"""

import argparse
import os

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with Unsloth LoRA/QLoRA")
    p.add_argument("--model_name", required=True, help="HF model ID or local path")
    p.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    p.add_argument("--output_dir", default="/workspace/outputs")
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
    p.add_argument("--load_in_4bit", type=str, default="true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_fraction", type=float, default=0.1)
    return p.parse_known_args()[0]


def format_conversations(example, tokenizer, max_seq_length):
    """Format conversation data for training. Supports both ShareGPT and OpenAI formats."""
    messages = []
    for t in example["conversations"]:
        if "from" in t:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            messages.append({"role": role_map.get(t["from"], t["from"]), "content": t["value"]})
        else:
            messages.append({"role": t.get("role", ""), "content": t["content"]})

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
    load_in_4bit = args.load_in_4bit.lower() in ("true", "1", "yes")

    # ---- Model + Tokenizer via Unsloth ----
    print(f"Loading model: {args.model_name} (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=load_in_4bit,
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

    # ---- LoRA via Unsloth ----
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=args.seed,
    )

    # ---- Dataset ----
    raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenized = raw_dataset.map(
        lambda ex: format_conversations(ex, tokenizer, args.max_seq_length),
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
        optim="adamw_8bit",
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
        skip_memory_metrics=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"Starting training: {len(train_dataset)} examples, "
          f"max_seq_length={args.max_seq_length}, batch_size={args.batch_size}")

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
    print(f"LoRA adapter saved to {args.output_dir}/lora_adapter")


if __name__ == "__main__":
    main()
