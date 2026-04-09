"""Merge a LoRA adapter into the base model and save the full merged model.

Usage:
    python merge.py --base_model google/gemma-4-e4b \
                    --adapter_path /workspace/outputs/job123/lora_adapter \
                    --output_dir /workspace/outputs/job123/merged
"""

import argparse
import gc
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Monkey-patch pynvml for DGX Spark GB10
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

    def _patched(handle, version=None):
        try:
            return _orig_nvmlDeviceGetMemoryInfo(handle, version) if version else _orig_nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError:
            return _FakeMemInfo()

    pynvml.nvmlDeviceGetMemoryInfo = _patched
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Flush NFS page cache after safetensors loads
# ---------------------------------------------------------------------------
try:
    import safetensors.torch
    _orig_load = safetensors.torch.load_file
    def _load_and_flush(filename, *args, **kwargs):
        result = _orig_load(filename, *args, **kwargs)
        try:
            fd = os.open(str(filename), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
            finally:
                os.close(fd)
        except (OSError, AttributeError):
            pass
        return result
    safetensors.torch.load_file = _load_and_flush
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", required=True, help="HF model ID or local path")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", required=True, help="Output path for merged model")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("Base model loaded.", flush=True)

    # Flush page cache after model load
    gc.collect()
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except PermissionError:
        pass

    print(f"Loading adapter: {args.adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Adapter loaded.", flush=True)

    print("Merging adapter into base model...", flush=True)
    model = model.merge_and_unload()
    print("Merge complete.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving merged model to: {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    print("Model saved.", flush=True)

    # Save tokenizer from adapter (it may have custom chat template)
    print("Saving tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
