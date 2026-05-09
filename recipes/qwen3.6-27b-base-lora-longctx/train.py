"""Fine-tune Qwen 3.6-27B BASE (dense, hybrid GatedDeltaNet + Gated-Attention,
multimodal) with DeepSpeed ZeRO-3 + LoRA.

Architecture notes (vs the 35B-A3B recipe this was adapted from):
- Dense, not MoE — no fused expert tensors. Drop `target_parameters`;
  rely on `target_modules=q/k/v/o_proj` only.
- Still hybrid: full-attention layers (16 of 64, every 4th: indices
  [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]) + GatedDeltaNet
  (linear-attention) layers. Same LoRA rule as the 35B's Mamba layers
  — only target the full-attention projections, never linear-attn
  in_proj / out_proj. Verified statically that q_proj only appears
  on the 16 full-attention LM layers, so PEFT's suffix matcher
  automatically skips GatedDeltaNet layers.
- Suffix matching also captures the MTP head's q/k/v/o_proj (one
  transformer block under `mtp.layers.0.*`) — neutralized below by
  the `frozen` loop's `mtp.` substring filter.
- Suffix matching also captures the vision tower's q/k/v/o_proj.
  Intentional: the recipe is kept multimodal-capable for future
  vision-aware fine-tunes, and matches the 35B-A3B baseline behavior
  for direct phase-a comparability.
- Multimodal wrapper present (vision tower + LM). Merge happens via
  scripts/merge_qwen3moe.py (its Case 1 dense 2D path). The generic
  scripts/merge.py would strip the multimodal wrapper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.patches import apply_all, flush_page_cache, fix_gemma4_use_cache
from lib.dataset import prepare_datasets
from lib.logging import setup_logging, LogMetricsCallback
from lib.tokenizer import setup_tokenizer
from lib.args import add_common_args, add_deepspeed_args

import argparse, gc, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

apply_all()

# Increase NCCL timeout BEFORE any process group init.
# ZeRO-3 loading does hundreds of broadcasts during from_pretrained;
# the default 30-min timeout is too short for 27B on DGX Spark.
import datetime
torch.distributed.constants.default_pg_timeout = datetime.timedelta(hours=4)
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(hours=4)


def main():
    p = argparse.ArgumentParser(description="Fine-tune Qwen 3.6-27B with DeepSpeed ZeRO-3 + LoRA")
    add_common_args(p)
    add_deepspeed_args(p)
    args = p.parse_known_args()[0]
    world_rank = int(os.environ.get("RANK", 0))

    setup_logging(args.output_dir)

    if args.ds_config:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        HfDeepSpeedConfig(args.ds_config)

    tokenizer = setup_tokenizer(args.model_name)

    train_ds, eval_ds = prepare_datasets(
        args.dataset, tokenizer, args.max_seq_length, args.eval_fraction, args.seed, world_rank)

    # Drop Gemma-only columns the dataset library always emits.
    qwen_drop_cols = [c for c in ("token_type_ids", "mm_token_type_ids") if c in train_ds.column_names]
    if qwen_drop_cols:
        train_ds = train_ds.remove_columns(qwen_drop_cols)
        if eval_ds is not None:
            eval_ds = eval_ds.remove_columns(qwen_drop_cols)

    print(f"[Rank {world_rank}] Loading model: {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[Rank {world_rank}] Model loaded.", flush=True)
    gc.collect()
    flush_page_cache()

    # Dense LoRA: just attention projections. PEFT's suffix matcher
    # automatically skips GatedDeltaNet layers (they don't expose
    # q_proj/k_proj/v_proj/o_proj names — verified in static check).
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))

    # Defensive freeze: catches MTP head LoRA (suffix matcher hits
    # mtp.layers.0.self_attn.{q,k,v,o}_proj). Also a no-op safety net
    # for any future Qwen variant exposing .gate.weight or router.
    frozen = 0
    for name, p in model.named_parameters():
        if ".gate.weight" in name or "router" in name or "mtp." in name:
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1

    if world_rank == 0:
        model.print_trainable_parameters()
        trainable, total = model.get_nb_trainable_parameters()
        pct = trainable / total * 100
        print(f"[LoRA capacity] {trainable:,} / {total:,} = {pct:.4f}%", flush=True)
        print(f"[Freeze] froze {frozen} router/mtp parameter tensors", flush=True)
        # Dense q/k/v/o LoRA on r=16 lands well under 0.5% (typically
        # 0.02-0.1% — varies with vision tower size). Fail loud only
        # if effectively nothing matched, which would mean the
        # q_proj/k_proj/v_proj/o_proj suffixes didn't hit any module.
        if pct < 0.001:
            raise RuntimeError(
                f"LoRA capacity suspiciously low ({pct:.4f}%): "
                f"target_modules likely didn't match any layer. Check "
                f"the model's parameter naming. Aborting."
            )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fix_gemma4_use_cache(model)  # Generic use_cache=True; safe no-op for non-Gemma archs.

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
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
            seed=args.seed, max_length=args.max_seq_length, packing=args.packing,
            report_to="none", deepspeed=args.ds_config, skip_memory_metrics=True,
            remove_unused_columns=False))

    if world_rank == 0:
        print(f"Starting training: {len(train_ds)} examples, "
              f"max_seq_length={args.max_seq_length}, batch_size={args.batch_size}", flush=True)

    gc.collect()
    flush_page_cache()

    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(f"{args.output_dir}/lora_adapter")
    if world_rank == 0:
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
