"""DGX Spark hardware workarounds.

Patches for NVIDIA GB10 unified memory architecture:
- pynvml: nvmlDeviceGetMemoryInfo not supported, return system RAM instead
- safetensors: flush NFS page cache after each shard load
- Gemma 4: unwrap ClippableLinear wrappers for PEFT compatibility
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


def flush_page_cache():
    """Drop system page cache via /proc. Requires root and writable /proc."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, OSError):
        return False


def unwrap_custom_linear(model):
    """Replace Gemma4ClippableLinear wrappers with standard nn.Linear for PEFT compatibility.

    Returns the number of modules replaced.
    """
    import torch.nn as nn
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear) and type(module).__name__ != 'Linear':
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], module.linear)
                replaced += 1
    if replaced:
        print(f"Replaced {replaced} custom linear wrappers with nn.Linear for LoRA", flush=True)
    return replaced


def fix_gemma4_use_cache(model):
    """Gemma 4 bug: use_cache=False corrupts attention outputs.

    gradient_checkpointing_enable sets use_cache=False by default.
    Force it back to True.
    """
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True


def apply_all():
    """Apply all DGX Spark patches. Call at the top of train.py before any imports."""
    patch_pynvml()
    patch_safetensors_cache()
