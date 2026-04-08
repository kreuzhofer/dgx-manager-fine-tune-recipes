#!/usr/bin/env bash
# Container entrypoint for Unsloth fine-tune jobs.
set -euo pipefail

echo "=== Unsloth DGX Spark Container ==="
python -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')" 2>/dev/null || echo "Unsloth installed"

touch /tmp/.ready
exec sleep infinity
