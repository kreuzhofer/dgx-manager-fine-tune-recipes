#!/usr/bin/env bash
# Launch Unsloth training (single-node, no torchrun needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

echo "=== DGX Manager Fine-Tune (Unsloth) ==="
echo "Node: $(hostname)"
echo "Script: ${TRAIN_SCRIPT}"
echo "Args: $@"
echo "========================================"

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")/../.." && pwd)/lib/setup_logging.sh"
setup_shell_log_tee "$@"

exec python "$TRAIN_SCRIPT" "$@"
