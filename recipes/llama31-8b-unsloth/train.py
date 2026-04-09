"""Fine-tune Llama 3.1 8B with Unsloth QLoRA."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.dataset import prepare_datasets
from lib.logging import setup_logging, LogMetricsCallback
from lib.args import add_common_args

import argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def main():
    p = argparse.ArgumentParser(description="Fine-tune with Unsloth QLoRA")
    add_common_args(p)
    p.add_argument("--load_in_4bit", type=str, default="true")
    args = p.parse_known_args()[0]
    load_in_4bit = args.load_in_4bit.lower() in ("true", "1", "yes")

    setup_logging(args.output_dir)

    print(f"Loading model: {args.model_name} (4bit={load_in_4bit})", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=load_in_4bit)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, 'chat_template', None):
        from lib.tokenizer import GEMMA_CHAT_TEMPLATE
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=args.lora_dropout,
        bias="none", use_gradient_checkpointing="unsloth", random_state=args.seed)

    train_ds, eval_ds = prepare_datasets(
        args.dataset, tokenizer, args.max_seq_length, args.eval_fraction, args.seed)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[LogMetricsCallback()],
        args=SFTConfig(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps, num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate, bf16=True, optim="adamw_8bit",
            warmup_steps=5, logging_steps=1, save_strategy="epoch",
            eval_strategy="epoch" if eval_ds else "no",
            seed=args.seed, max_length=args.max_seq_length, packing=False,
            report_to="none", skip_memory_metrics=True))

    print(f"Starting training: {len(train_ds)} examples", flush=True)
    trainer.train()

    model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
    print(f"LoRA adapter saved to {args.output_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
