"""Qwen 3.5/3.6 MoE-aware LoRA merge — bypasses PEFT's merge_and_unload + vLLM's runtime LoRA.

Why this script exists:

Qwen 3.5/3.6 MoE models are released as `Qwen3_5MoeForConditionalGeneration`
(multimodal wrapper) on top of an `Qwen3_5MoeForCausalLM` LM. PEFT's
`merge_and_unload()` + `save_pretrained()` strips the multimodal wrapper and
saves the LM with `model_type=qwen3_5_moe_text` — a leaf-config that
transformers/vLLM doesn't recognize. The Gemma-style merge.py also adds
`fix_clippable_linear_keys` which copies "all missing keys from base", which
duplicates lm_head and confuses vLLM's loader.

vLLM's runtime `--enable-lora --lora-modules` route doesn't help either: it
only accepts adapter keys that match its native MoE LoRA naming
(`experts.lora_*`), and it silently rejects the half saved as
`experts.base_layer.lora_*` (the trained gate_up_proj deltas).

This script sidesteps both:
- Reads the PEFT adapter directly from safetensors (no PEFT runtime needed).
- Adds LoRA deltas into the BASE model's safetensors files in-place (writes
  them to a new output dir but with identical layout/keys/config to the base).
- Output dir is byte-compatible with the base's serving path — vLLM dispatches
  it the same way as the unmerged base, no config gymnastics required.

PEFT block-diagonal LoRA layout for `target_parameters=["experts.gate_up_proj",
"experts.down_proj"]`:

  Note: PEFT names these inconsistently (one with `.base_layer.` infix and one
  without). Empirically observed for Qwen3.5/3.6 MoE adapters from peft 0.19:

    experts.base_layer.lora_A.weight  shape [E*r, hidden]            → gate_up_proj
    experts.base_layer.lora_B.weight  shape [2*moe_inter, E*r]
    experts.lora_A.weight             shape [E*r, moe_inter]         → down_proj
    experts.lora_B.weight             shape [hidden, E*r]

  Per-expert delta: A_e = A[e*r:(e+1)*r, :], B_e = B[:, e*r:(e+1)*r]
                    delta_e = (B_e @ A_e) * (alpha / r)

  For gate_up_proj the delta has shape [2*moe_inter, hidden]; the base
  per-expert tensor shape is [2*moe_inter, hidden] so we add directly.
  For down_proj the delta has shape [hidden, moe_inter]; matches base.

For attention LoRA on `q_proj`/`k_proj`/`v_proj`/`o_proj` the merge is the
standard B@A * (alpha/r) added into the base Linear weight.

Usage:
  python merge_qwen3moe.py \\
      --base /mnt/tank/models/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev> \\
      --adapter /mnt/tank/outputs/<jobId>/lora_adapter \\
      --output /mnt/tank/outputs/<jobId>/merged-clean
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _resolve_base_dir(base):
    """Accept either a local snapshot dir or a HuggingFace model id and
    return a Path pointing at the actual snapshot directory (with
    config.json + *.safetensors)."""
    p = Path(base)
    if p.is_dir():
        return p
    # Looks like an HF id (e.g. "Qwen/Qwen3.6-35B-A3B"): resolve to the
    # cached snapshot. local_files_only=True avoids any download; the
    # merge container has HF_HOME mounted from the shared cache.
    if "/" in base and not base.startswith("/"):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(base, local_files_only=True))
    raise SystemExit(f"Cannot resolve base model: {base} (not a dir, not a HF id)")


def load_st_index(base_dir):
    """Load model.safetensors.index.json → {tensor_name: shard_filename}."""
    idx = Path(base_dir) / "model.safetensors.index.json"
    if not idx.exists():
        # Single-file model
        candidates = list(Path(base_dir).glob("*.safetensors"))
        if len(candidates) != 1:
            raise SystemExit(f"No index.json and {len(candidates)} safetensors files in {base_dir}")
        return {}, [candidates[0].name]
    data = json.loads(idx.read_text())
    weight_map = data["weight_map"]
    files = sorted(set(weight_map.values()))
    return weight_map, files


def safe_open_all(base_dir, files):
    """Open all base safetensors. Return dict shard_name → safe_open handle."""
    return {name: safe_open(str(Path(base_dir) / name), framework="pt", device="cpu") for name in files}


def load_adapter(adapter_dir):
    """Read PEFT adapter config + safetensors. Return (config, tensors_dict)."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    af = Path(adapter_dir) / "adapter_model.safetensors"
    tensors = {}
    with safe_open(str(af), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return cfg, tensors


def strip_peft_prefix(k):
    """Strip PEFT's `base_model.model.` prefix and `.base_layer.` infix.
    Returns the bare base-model parameter path the LoRA targets."""
    if k.startswith("base_model.model."):
        k = k[len("base_model.model."):]
    return k


def find_lora_pairs(adapter_tensors):
    """Group adapter keys into LoRA pairs and resolve their target parameter
    name in the BASE model (multimodal-wrapped Qwen 3.6).

    Adapter keys look like:
      base_model.model.model.layers.N.mlp.experts.base_layer.lora_A.weight
      base_model.model.model.layers.N.mlp.experts.lora_A.weight        (bare)
      base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight

    Base safetensors keys look like (multimodal):
      model.language_model.layers.N.mlp.experts.gate_up_proj            (3D fused)
      model.language_model.layers.N.mlp.experts.down_proj
      model.language_model.layers.N.self_attn.q_proj.weight             (Linear)

    Translation rules (applied here once so the merge loop does straight matches):
      - Strip leading `base_model.model.` (PEFT wrap prefix).
      - Strip the `.lora_A.weight` / `.lora_B.weight` suffix.
      - For experts pairs, slot tells which fused tensor to target:
          slot=base_layer → root_clean is `model.layers.N.mlp.experts`;
                            base param is `model.language_model.layers.N.mlp.experts.gate_up_proj`
          slot=bare       → same root_clean; base param is `.down_proj`
      - For attention pairs, root_clean is `model.layers.N.self_attn.q_proj`;
                              base param is `model.language_model.layers.N.self_attn.q_proj.weight`
      - In all cases: replace `model.layers.` with `model.language_model.layers.`
        (Qwen 3.6's multimodal wrapper puts the LM under language_model).
    """
    by_root = {}  # root_clean → {slot → {'A': T, 'B': T}}
    for k, v in adapter_tensors.items():
        if not k.endswith(".lora_A.weight") and not k.endswith(".lora_B.weight"):
            continue
        ab = "A" if k.endswith(".lora_A.weight") else "B"
        bare = strip_peft_prefix(k)
        root = bare[:-len(f".lora_{ab}.weight")]
        is_base_layer = root.endswith(".base_layer")
        root_clean = root[:-len(".base_layer")] if is_base_layer else root
        slot = "base_layer" if is_base_layer else "bare"
        by_root.setdefault(root_clean, {}).setdefault(slot, {})[ab] = v

    pairs = []
    for root, slots in by_root.items():
        for slot, ab in slots.items():
            if "A" not in ab or "B" not in ab:
                continue
            # Resolve to base parameter name
            target = _resolve_target(root, slot)
            if target is None:
                continue  # unknown root pattern; surface in summary later
            pairs.append({
                "target_root": root,
                "slot": slot,
                "target_param": target,
                "lora_A": ab["A"],
                "lora_B": ab["B"],
            })
    return pairs


def _resolve_target(root_clean, slot):
    """Map adapter root + slot to the BASE multimodal parameter name."""
    # Add multimodal LM prefix
    if root_clean.startswith("model.layers."):
        base_root = "model.language_model.layers." + root_clean[len("model.layers."):]
    else:
        # Unknown adapter prefix — leave as-is and hope for the best
        base_root = root_clean
    # Decide suffix
    if base_root.endswith(".mlp.experts"):
        # Slot determines which fused tensor:
        #   base_layer → gate_up_proj, bare → down_proj
        return base_root + (".gate_up_proj" if slot == "base_layer" else ".down_proj")
    if any(base_root.endswith(f".self_attn.{p}") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return base_root + ".weight"
    # Fallback
    return base_root + ".weight"


def main():
    p = argparse.ArgumentParser()
    # Accept both --base/--adapter/--output (original) and --base_model/
    # --adapter_path/--output_dir (agent merge contract, shared with
    # scripts/merge.py). The agent passes the second form; either works
    # at the command line.
    p.add_argument("--base", "--base_model", dest="base", required=True,
                   help="Base model: local snapshot dir OR HuggingFace id (e.g. Qwen/Qwen3.6-35B-A3B)")
    p.add_argument("--adapter", "--adapter_path", dest="adapter", required=True, help="PEFT adapter dir")
    p.add_argument("--output", "--output_dir", dest="output", required=True, help="Output dir for merged safetensors")
    p.add_argument("--num-experts", type=int, default=256, help="MoE expert count (Qwen 3.6: 256)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    base_dir = _resolve_base_dir(args.base)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading adapter config from {adapter_dir}/adapter_config.json")
    adapter_cfg, adapter_tensors = load_adapter(adapter_dir)
    r = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    scaling = alpha / r
    print(f"  rank={r}, alpha={alpha}, scaling={scaling}")
    print(f"  adapter has {len(adapter_tensors)} tensors")

    pairs = find_lora_pairs(adapter_tensors)
    print(f"Found {len(pairs)} LoRA pairs to merge")

    # Index pairs by their resolved BASE parameter name for direct lookup.
    pairs_by_target = {}
    for p_ in pairs:
        pairs_by_target.setdefault(p_["target_param"], []).append(p_)
    print("Sample resolved targets:")
    for tp in list(pairs_by_target.keys())[:5]:
        print(f"  {tp}")

    # Load base index + open all shards
    print(f"Loading base safetensors index from {base_dir}")
    weight_map, files = load_st_index(base_dir)
    if not weight_map:
        raise SystemExit("Single-file safetensors not supported by this script (Qwen 3.6 base is sharded)")
    print(f"  base has {len(weight_map)} tensors across {len(files)} shards")

    # We process one shard at a time: load all tensors, apply any LoRA deltas
    # whose target lives in this shard, save to output. Memory-efficient.
    merged_count = 0
    applied_targets = set()

    for shard in files:
        src = base_dir / shard
        dst = out_dir / shard
        print(f"\n=== Shard {shard} ===")
        with safe_open(str(src), framework="pt", device="cpu") as f:
            shard_tensors = {k: f.get_tensor(k) for k in f.keys()}
        applied_in_shard = 0

        for tname, tensor in list(shard_tensors.items()):
            if tname not in pairs_by_target:
                continue
            # There should normally be exactly one pair per target. If multiple
            # are registered (shouldn't happen with sane PEFT output), apply all.
            for p_ in pairs_by_target[tname]:
                delta = compute_delta(p_, tensor, args.num_experts, scaling)
                if delta is None:
                    print(f"  ! could not compute delta for {tname} (slot={p_['slot']}, "
                          f"A={tuple(p_['lora_A'].shape)}, B={tuple(p_['lora_B'].shape)}, "
                          f"base={tuple(tensor.shape)})")
                    continue
                if delta.shape != tensor.shape:
                    print(f"  ! shape mismatch for {tname}: base {tuple(tensor.shape)} vs delta {tuple(delta.shape)}")
                    continue
                tensor = (tensor.to(torch.float32) + delta.to(torch.float32)).to(out_dtype)
                shard_tensors[tname] = tensor
                applied_in_shard += 1
                merged_count += 1
                print(f"  + merged into {tname:<78} (slot={p_['slot']}, |delta|={delta.norm().item():.4f})")
            applied_targets.add(tname)

        # Cast all surviving tensors to out_dtype to keep file homogeneous
        for k in list(shard_tensors.keys()):
            if shard_tensors[k].dtype != out_dtype:
                shard_tensors[k] = shard_tensors[k].to(out_dtype)
        save_file(shard_tensors, str(dst), metadata={"format": "pt"})
        print(f"  wrote {dst} ({applied_in_shard} pairs applied, {len(shard_tensors)} tensors)")

    # Copy non-safetensors files (config, tokenizer, etc.) from base
    print(f"\nCopying config/tokenizer files from {base_dir}")
    for f in os.listdir(base_dir):
        src = base_dir / f
        if not src.is_file():
            continue
        if f.endswith(".safetensors"):
            continue
        dst = out_dir / f
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  copied {f}")

    # Report
    print()
    print(f"=== Done ===")
    print(f"  Merged {merged_count} / {len(pairs)} LoRA pairs")
    unapplied = [p_ for p_ in pairs if p_["target_param"] not in applied_targets]
    if unapplied:
        print(f"  WARNING: {len(unapplied)} pairs not applied (target tensor not found):")
        for p_ in unapplied[:10]:
            print(f"    target={p_['target_param']}  (root={p_['target_root']}, slot={p_['slot']})")
        if len(unapplied) > 10:
            print(f"    ... and {len(unapplied) - 10} more")
    print(f"  Output: {out_dir}")


def compute_delta(pair, base_tensor, num_experts, scaling):
    """Compute LoRA delta for one pair, in the shape of base_tensor.

    Two patterns:
    1. Standard 2D LoRA on a Linear weight (q/k/v/o_proj):
         A: [r, in], B: [out, r], delta = B @ A scaled, shape [out, in].
    2. Block-diagonal LoRA on a fused 3D MoE expert tensor
       (experts.gate_up_proj or experts.down_proj):
         Base shape [E, out_per_expert, in_per_expert].
         A: [E*r, in_per_expert]   (PEFT concat across experts)
         B: [out_per_expert, E*r]
         Per expert e:
           A_e = A[e*r:(e+1)*r, :]  shape [r, in]
           B_e = B[:, e*r:(e+1)*r]  shape [out, r]
           delta_e = B_e @ A_e      shape [out, in]
         Stack into [E, out, in] then add to base.
    """
    A = pair["lora_A"].to(torch.float32)
    B = pair["lora_B"].to(torch.float32)

    # Case 1: standard 2D Linear LoRA
    if base_tensor.ndim == 2:
        # A: [r, in_features], B: [out_features, r]
        if A.shape[1] == base_tensor.shape[1] and B.shape[0] == base_tensor.shape[0] and A.shape[0] == B.shape[1]:
            delta = (B @ A) * scaling
            return delta
        return None

    # Case 2: fused 3D expert tensor [E, out_per_expert, in_per_expert]
    if base_tensor.ndim == 3:
        E, out_pe, in_pe = base_tensor.shape
        if E != num_experts:
            return None
        # A should be [E*r, in_pe], B should be [out_pe, E*r]
        if A.shape[1] != in_pe or B.shape[0] != out_pe:
            return None
        if A.shape[0] != B.shape[1]:
            return None
        Er = A.shape[0]
        if Er % E != 0:
            return None
        r = Er // E
        # Split into per-expert blocks and compute deltas
        delta = torch.empty((E, out_pe, in_pe), dtype=torch.float32)
        for e in range(E):
            A_e = A[e * r:(e + 1) * r, :]   # [r, in_pe]
            B_e = B[:, e * r:(e + 1) * r]   # [out_pe, r]
            delta[e] = B_e @ A_e
        delta *= scaling
        return delta

    return None


if __name__ == "__main__":
    main()
