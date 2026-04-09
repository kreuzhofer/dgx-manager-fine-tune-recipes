"""Merge a LoRA adapter into the base model and save the full merged model.

Usage:
    python merge.py --base_model google/gemma-4-e4b \
                    --adapter_path /workspace/outputs/job123/lora_adapter \
                    --output_dir /workspace/outputs/job123/merged
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.patches import apply_all, unwrap_custom_linear, flush_page_cache
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

    unwrap_custom_linear(model)

    print(f"Loading adapter: {args.adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Adapter loaded.", flush=True)

    print("Merging adapter into base model...", flush=True)
    model = model.merge_and_unload()
    print("Merge complete.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to: {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    print("Saving tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
