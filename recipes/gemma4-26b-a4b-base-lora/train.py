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

    # MoE LoRA: standard target_modules cannot reach the fused-expert
    # `mlp.experts.gate_up_proj` / `down_proj` Parameter tensors (they're not
    # nn.Linear modules, so PEFT's named_modules() walk skips them). Without
    # target_parameters we only LoRA the attention projections and end up with
    # ~30M / 26B = 0.12% trainable — way below the normal ~1% range — and the
    # experts (where most knowledge lives) stay frozen. target_parameters is
    # PEFT's mechanism for slicing into those fused tensors per-expert.
    attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    moe_param_targets = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=attn_targets,
        target_parameters=moe_param_targets,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))

    # Freeze the router — fine-tuning routing weights destabilizes which
    # experts see which tokens, creating a feedback loop with the LoRA updates.
    router_frozen = 0
    for name, p in model.named_parameters():
        if ".gate.weight" in name or "router" in name:
            if p.requires_grad:
                p.requires_grad = False
                router_frozen += 1

    if world_rank == 0:
        model.print_trainable_parameters()
        trainable, total = model.get_nb_trainable_parameters()
        pct = trainable / total * 100
        print(f"[LoRA capacity] {trainable:,} / {total:,} = {pct:.3f}%", flush=True)
        print(f"[Router] froze {router_frozen} parameter tensors", flush=True)
        # Sanity check: with target_parameters reaching 256 experts × rank 16
        # we expect ~1% trainable. If we see ~0.1% the experts weren't reached
        # and there's no point burning the rest of the run.
        if pct < 0.5:
            raise RuntimeError(
                f"LoRA capacity too low ({pct:.3f}%): target_parameters likely "
                f"failed to reach the fused MoE experts. Check PEFT version "
                f"(needs >= 0.16) and parameter names in the model. Aborting."
            )

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
            save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=args.save_total_limit, save_only_model=args.save_only_model,
            eval_strategy="steps" if eval_ds else "no", eval_steps=args.eval_steps,
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

    # Resume from checkpoint if requested. Pass `True` to auto-find the latest
    # checkpoint-* dir under output_dir; pass a path to use a specific checkpoint.
    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    # All ranks must call save_model with ZeRO-3 (parameters are distributed)
    trainer.save_model(f"{args.output_dir}/lora_adapter")
    if world_rank == 0:
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
