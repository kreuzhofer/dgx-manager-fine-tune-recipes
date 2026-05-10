"""Fine-tune Qwen 3.6-27B BASE (dense, hybrid GatedDeltaNet + Gated-Attention,
multimodal) with DeepSpeed ZeRO-3 + LoRA. Attention-only LoRA targets.

GB10 unified-memory mitigations (shared with the attn-mlp sibling,
validated end-to-end at seq=8192):

1. Liger-Kernel fused linear cross-entropy. Skips materializing the
   [batch, seq, vocab] logits tensor at the loss layer — at Qwen
   3.6's 248k vocab + seq=8192 that's a multi-GB transient. ~30%
   per-step training speedup with bit-identical loss. Only the
   loss-layer fusion is enabled; RoPE / RMSNorm / SwiGLU patches
   are left off because Qwen 3.6's hybrid GatedDeltaNet+attention
   is not validated against Liger's other Qwen3 kernels.

2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in launch.sh —
   reduces fragmentation in the CUDA caching allocator under
   unified-memory pressure.

3. per_device_eval_batch_size=1 — load-bearing. HF defaults eval
   batch to 8 which makes the eval logits tensor 8× larger than
   training; with stock CE that's ~65 GB after .float() cast and
   OOMs even when training fits. Liger's lce_forward only patches
   the training-mode forward, so eval still uses stock CE — the
   pin keeps eval memory == training memory.

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


def apply_liger_class_level_patches() -> set:
    """Apply every Liger Qwen3-family class-level patcher we know about.

    IMPORTANT: liger's `apply_liger_kernel_to_<arch>` functions monkey-patch
    `transformers.models.<arch>.*` classes. They silently no-op when their
    target arch module isn't loaded, so a patcher returning without raising
    does NOT mean the loaded model's class was actually touched. We run all
    candidates that exist (no early-exit) and rely on `verify_and_force_patch`
    after model load to confirm and, if necessary, attach lce_forward
    directly to the loaded class.

    Only `fused_linear_cross_entropy` is enabled — every other liger swap is
    explicitly disabled to keep Qwen 3.6's hybrid GatedDeltaNet+attention
    path stock."""
    try:
        from liger_kernel.transformers import monkey_patch as lk
    except ImportError as e:
        print(f"[Liger] liger-kernel import failed ({e}); using stock loss", flush=True)
        return set()
    candidates = (
        "apply_liger_kernel_to_qwen3_5",
        "apply_liger_kernel_to_qwen3_next",
        "apply_liger_kernel_to_qwen3",
        "apply_liger_kernel_to_qwen2",
    )
    kwargs = dict(rope=False, swiglu=False, cross_entropy=False,
                  fused_linear_cross_entropy=True, rms_norm=False)
    applied = set()
    for name in candidates:
        fn = getattr(lk, name, None)
        if fn is None:
            continue
        try:
            fn(**kwargs)
            applied.add(name)
        except TypeError:
            try:
                fn(fused_linear_cross_entropy=True)
                applied.add(name + "(min-kwargs)")
            except Exception as e:
                print(f"[Liger] {name} rejected kwargs: {e}", flush=True)
        except Exception as e:
            print(f"[Liger] {name} failed: {e}", flush=True)
    print(f"[Liger] class-level patchers ran: {sorted(applied) or 'none'}", flush=True)
    return applied


def verify_and_force_patch(model) -> str:
    """Check whether the loaded model's forward is a Liger lce_forward.
    If the class-level patchers missed (e.g. they patched qwen3_next while
    the loaded model is qwen3_5), import lce_forward from the most-specific
    arch-matching liger module and assign it to the loaded class directly."""
    cls = type(model)
    fwd_module = (getattr(cls.forward, "__module__", "") or "").lower()
    if "liger" in fwd_module:
        return "class-level-hit"
    try:
        from importlib import import_module
    except ImportError:
        return "no-liger"
    sources = (
        "liger_kernel.transformers.model.qwen3_5",
        "liger_kernel.transformers.model.qwen3_next",
        "liger_kernel.transformers.model.qwen3",
        "liger_kernel.transformers.model.qwen2",
    )
    for src in sources:
        try:
            mod = import_module(src)
        except ImportError:
            continue
        lce_fn = getattr(mod, "lce_forward", None)
        if lce_fn is None:
            continue
        cls.forward = lce_fn
        new_module = (getattr(cls.forward, "__module__", "") or "").lower()
        if "liger" in new_module:
            print(f"[Liger] Manually attached {src}.lce_forward to {cls.__name__}", flush=True)
            return "manual-patched"
    return "no-lce-source"


_liger_class_patches = apply_liger_class_level_patches()


class LigerSFTTrainer(SFTTrainer):
    """SFTTrainer compatible with Liger fused linear cross-entropy.

    Stock SFTTrainer.compute_loss does its own logits-shift + CE on
    `outputs.logits`, which crashes with `'NoneType' object is not
    subscriptable` because liger's lce_forward correctly returns
    logits=None when labels are provided. Override: when the model
    already returned a loss (liger active), use it directly.
    Otherwise fall back to stock behavior."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
            inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


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

    # Verify the Liger class-level patch actually hit the loaded class. Run on
    # every rank because forward replacement is a class attribute change that
    # all ranks need.
    klass = type(model).__name__
    module = type(model).__module__
    patch_status = verify_and_force_patch(model)
    if world_rank == 0:
        print(f"[Liger] Loaded model class: {module}.{klass}", flush=True)
        print(f"[Liger] Class-level patchers ran: {sorted(_liger_class_patches) or 'none'}", flush=True)
        print(f"[Liger] Final patch status: {patch_status} "
              f"(forward.__module__={getattr(type(model).forward, '__module__', '?')})", flush=True)

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

    trainer = LigerSFTTrainer(
        model=model, processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[LogMetricsCallback()],
        args=SFTConfig(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            # HF defaults per_device_eval_batch_size to 8 — at seq=8192 with
            # Qwen 3.6's 248k vocab, the eval forward materializes a
            # [8, 8192, 248k] bf16 logits tensor (~32 GB) and ForCausalLMLoss
            # casts to fp32 doubling it (~65 GB). On single-rank GB10 that
            # OOMs after step 5. Pin to 1 to match training memory profile.
            per_device_eval_batch_size=1,
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
