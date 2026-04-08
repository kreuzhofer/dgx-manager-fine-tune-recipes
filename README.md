# DGX Manager Fine-Tune Recipes

Training recipes for [DGX Manager](https://github.com/kreuzhofer/dgx-manager). Each recipe is a self-contained directory with a training script, launch script, container entrypoint, and configuration.

## Structure

```
recipes/
└── gemma4-e2b-lora/
    ├── recipe.yaml      # Metadata, defaults, hardware requirements
    ├── train.py         # Training script
    ├── launch.sh        # Launcher (torchrun wrapper)
    ├── entrypoint.sh    # Container setup (pip deps)
    └── ds_config.json   # DeepSpeed config
```

## Recipe Format

Each recipe directory must contain a `recipe.yaml` with:

- `name`, `description` — human-readable metadata
- `base_model` — HuggingFace model ID
- `framework` — `deepspeed`, `fsdp2`, `unsloth`, or `torchtune`
- `method` — `lora` or `full`
- `container.image` — Docker image to use
- `scripts` — relative paths to entrypoint, train, launch scripts
- `defaults` — default hyperparameters (overridable from the dashboard)
- `hardware` — minimum nodes, GPUs per node, estimated VRAM

## Usage

Recipes are discovered automatically by the DGX Manager agent. Point the agent to this repo via the `TRAINING_REPO_PATH` environment variable.
