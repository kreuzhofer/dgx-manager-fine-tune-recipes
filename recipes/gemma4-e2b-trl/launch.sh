#!/usr/bin/env bash
# Launch plain TRL training (single GPU, no torchrun/DeepSpeed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

# Flush page cache
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

echo "=== DGX Manager Fine-Tune (TRL) ==="
echo "Node: $(hostname)"
echo "Script: ${TRAIN_SCRIPT}"
echo "Args: $@"
echo "====================================="

exec python "$TRAIN_SCRIPT" "$@"
