"""DGX Spark hardware workarounds and model compatibility patches.

- pynvml: nvmlDeviceGetMemoryInfo not supported on GB10, return system RAM
- safetensors: flush NFS page cache after each shard load
- PEFT: teach LoRA to handle Gemma4ClippableLinear without unwrapping
"""

import os


def patch_pynvml():
    """Monkey-patch pynvml for DGX Spark GB10 which doesn't support some NVML calls."""
    try:
        import pynvml
        _orig = pynvml.nvmlDeviceGetMemoryInfo

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
                return _orig(handle, version) if version else _orig(handle)
            except pynvml.NVMLError:
                return _FakeMemInfo()

        pynvml.nvmlDeviceGetMemoryInfo = _patched
    except ImportError:
        pass


def patch_safetensors_cache():
    """Flush NFS page cache after each safetensors shard load."""
    try:
        import safetensors.torch
        _orig = safetensors.torch.load_file

        def _load_and_flush(filename, *args, **kwargs):
            result = _orig(filename, *args, **kwargs)
            try:
                fd = os.open(str(filename), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)  # POSIX_FADV_DONTNEED
                finally:
                    os.close(fd)
            except (OSError, AttributeError):
                pass
            return result

        safetensors.torch.load_file = _load_and_flush
    except ImportError:
        pass


def patch_peft_for_clippable_linear():
    """Teach PEFT's LoRA to handle Gemma4ClippableLinear by targeting its inner nn.Linear.

    Instead of modifying the model architecture (which breaks weight key names for serving),
    we patch PEFT's dispatch to recognize ClippableLinear and wrap its inner .linear layer.
    The model architecture and weight names stay unchanged.
    """
    try:
        import torch.nn as nn
        from peft.tuners.lora import model as lora_model
        from peft.tuners.lora.layer import Linear as LoraLinear

        _orig_dispatch = lora_model.dispatch_default

        def _patched_dispatch(target, adapter_name, lora_config, **kwargs):
            # If the target has a .linear attribute that is nn.Linear,
            # it's a wrapper like Gemma4ClippableLinear — target the inner linear
            if (hasattr(target, 'linear') and isinstance(target.linear, nn.Linear)
                    and not isinstance(target, nn.Linear)):
                kwargs.update(lora_config.loftq_config)
                return LoraLinear(target.linear, adapter_name, **kwargs)
            return _orig_dispatch(target, adapter_name, lora_config, **kwargs)

        lora_model.dispatch_default = _patched_dispatch
        print("Patched PEFT dispatch for Gemma4ClippableLinear support", flush=True)
    except ImportError:
        pass


def flush_page_cache():
    """Drop system page cache via /proc. Requires root and writable /proc."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, OSError):
        return False


def fix_gemma4_use_cache(model):
    """Gemma 4 bug: use_cache=False corrupts attention outputs.

    gradient_checkpointing_enable sets use_cache=False by default.
    Force it back to True.
    """
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True


def apply_all():
    """Apply all DGX Spark patches. Call at the top of train.py before any model imports."""
    patch_pynvml()
    patch_safetensors_cache()
    patch_peft_for_clippable_linear()
