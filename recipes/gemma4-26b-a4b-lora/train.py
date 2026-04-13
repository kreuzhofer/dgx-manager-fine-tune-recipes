"""Fine-tune Gemma 4 E4B with DeepSpeed ZeRO + LoRA."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.patches import apply_all, flush_page_cache, fix_gemma4_use_cache
from lib.dataset import prepare_datasets, Gemma4DataCollator
from lib.logging import setup_logging, LogMetricsCallback
from lib.tokenizer import setup_tokenizer
from lib.args import add_common_args, add_deepspeed_args

import argparse, gc, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

apply_all()

# Increase NCCL timeout BEFORE any process group init.
# ZeRO-3 loading does hundreds of broadcasts during from_pretrained;
# the default 30-min timeout is too short for 26B on DGX Spark.
import datetime
torch.distributed.constants.default_pg_timeout = datetime.timedelta(hours=4)
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(hours=4)


def main():
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 26B-A4B with DeepSpeed ZeRO-3 + LoRA")
    add_common_args(p)
    add_deepspeed_args(p)
    args = p.parse_known_args()[0]
    world_rank = int(os.environ.get("RANK", 0))

    setup_logging(args.output_dir)

    if args.ds_config:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        HfDeepSpeedConfig(args.ds_config)

    tokenizer = setup_tokenizer(args.model_name)

    # Tokenize dataset BEFORE loading model to avoid fork OOM
    # (num_proc=4 in map() forks child processes — if model is already
    # loaded, each fork inherits 50GB+ virtual memory, exceeding 119GB)
    train_ds, eval_ds = prepare_datasets(
        args.dataset, tokenizer, args.max_seq_length, args.eval_fraction, args.seed, world_rank)

    print(f"[Rank {world_rank}] Loading model: {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[Rank {world_rank}] Model loaded.", flush=True)
    gc.collect()
    flush_page_cache()

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    if world_rank == 0:
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fix_gemma4_use_cache(model)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        data_collator=Gemma4DataCollator(tokenizer=tokenizer, mlm=False),
        train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[LogMetricsCallback()],
        args=SFTConfig(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps, num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate, bf16=True, optim="adamw_torch",
            warmup_steps=5, logging_steps=1,
            save_strategy="steps", save_steps=500, save_total_limit=3,
            eval_strategy="epoch" if eval_ds else "no",
            seed=args.seed, max_length=args.max_seq_length, packing=False,
            report_to="none", deepspeed=args.ds_config, skip_memory_metrics=True,
            remove_unused_columns=False))

    if world_rank == 0:
        print(f"Starting training: {len(train_ds)} examples, "
              f"max_seq_length={args.max_seq_length}, batch_size={args.batch_size}", flush=True)

    # Critical: flush NFS page cache before first training step
    # ZeRO-3 all-gather + stale page cache can exceed DGX Spark's
    # unified memory (GPU+CPU share 128GB)
    gc.collect()
    flush_page_cache()

    trainer.train()

    # All ranks must call save_model with ZeRO-3 (parameters are distributed)
    trainer.save_model(f"{args.output_dir}/lora_adapter")
    if world_rank == 0:
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
