#!/usr/bin/env bash
# Container entrypoint for Unsloth fine-tune jobs.
# Upgrades unsloth + transformers for Gemma 4 support (image ships with older versions).
set -euo pipefail

echo "=== Unsloth DGX Spark Container ==="

echo "Upgrading unsloth + transformers for Gemma 4 support..."
pip install -q --break-system-packages --upgrade unsloth unsloth-zoo transformers 2>&1 | tail -3

python -c "import unsloth; print(f'Unsloth: {unsloth.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}')"

echo "=== Ready ==="
touch /tmp/.ready
exec sleep infinity
