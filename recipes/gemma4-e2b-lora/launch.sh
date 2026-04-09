#!/usr/bin/env bash
# Launch single-node DeepSpeed training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
DS_CONFIG="${SCRIPT_DIR}/ds_config.json"

# Find an available master port (avoid conflicts with other training jobs)
MASTER_PORT=9901
while ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} " || lsof -i :${MASTER_PORT} >/dev/null 2>&1; do
    MASTER_PORT=$((MASTER_PORT + 1))
done

# Flush page cache
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

echo "=== DGX Manager Fine-Tune ==="
echo "Node: $(hostname)"
echo "Script: ${TRAIN_SCRIPT}"
echo "DeepSpeed config: ${DS_CONFIG}"
echo "Master port: ${MASTER_PORT}"
echo "Args: $@"
echo "============================="

exec torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --ds_config "$DS_CONFIG" \
    "$@"
