"""Merge a LoRA adapter into the base model and save the full merged model.

Usage:
    python merge.py --base_model google/gemma-4-e4b \
                    --adapter_path /workspace/outputs/job123/lora_adapter \
                    --output_dir /workspace/outputs/job123/merged
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.patches import apply_all, flush_page_cache, fix_clippable_linear_keys
from lib.logging import setup_logging

import argparse, gc, torch

apply_all()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    setup_logging(args.output_dir, filename="merge.log")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True)
    print("Base model loaded.", flush=True)
    gc.collect()
    flush_page_cache()

    # Note: do NOT unwrap ClippableLinear here — that would cause a weight name
    # mismatch when vLLM loads the saved model. Instead, unwrap only for PEFT
    # compatibility, then re-wrap after merge if needed.

    print(f"Loading adapter: {args.adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Adapter loaded.", flush=True)

    print("Merging adapter into base model...", flush=True)
    model = model.merge_and_unload()
    print("Merge complete.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to: {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Fix ClippableLinear weight key names for vLLM compatibility
    print("Fixing weight keys for ClippableLinear modules...", flush=True)
    fix_clippable_linear_keys(args.output_dir, args.base_model)

    print("Saving tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Copy additional config files from base model that vLLM needs
    # (preprocessor_config.json, etc. for multimodal models)
    print("Copying config files from base model...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        import shutil
        base_path = snapshot_download(args.base_model, allow_patterns=["*.json", "*.jinja"])
        for fname in os.listdir(base_path):
            if fname.endswith(".json") or fname.endswith(".jinja"):
                dst = os.path.join(args.output_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(base_path, fname), dst)
                    print(f"  Copied {fname}", flush=True)
    except Exception as e:
        print(f"  Warning: could not copy base model configs: {e}", flush=True)

    print(f"Merged model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
