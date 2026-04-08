#!/usr/bin/env bash
# Launch single-node DeepSpeed training.
#
# Usage:
#   /workspace/src/github/dgx-manager-fine-tune-recipes/recipes/gemma4-e2b-lora/launch.sh [args...]
#
# Examples:
#   launch.sh --model_name google/gemma-4-e2b --dataset /workspace/data/my-data.jsonl
#   launch.sh --model_name google/gemma-4-e2b --dataset /workspace/data/my-data.jsonl --max_steps 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
DS_CONFIG="${SCRIPT_DIR}/ds_config.json"

# Flush page cache before training to maximize available RAM
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

echo "=== DGX Manager Fine-Tune ==="
echo "Node: $(hostname)"
echo "Script: ${TRAIN_SCRIPT}"
echo "DeepSpeed config: ${DS_CONFIG}"
echo "Args: $@"
echo "============================="

exec torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=9901 \
    "$TRAIN_SCRIPT" \
    --ds_config "$DS_CONFIG" \
    "$@"
