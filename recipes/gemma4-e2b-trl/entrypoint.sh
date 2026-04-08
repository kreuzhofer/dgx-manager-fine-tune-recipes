#!/usr/bin/env bash
# Container entrypoint — install Python deps and signal readiness.
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q \
    "transformers>=4.51.0" \
    peft \
    datasets \
    "trl>=0.16.0" \
    accelerate \
    hf_transfer

echo "=== Ready ==="
touch /tmp/.ready
exec sleep infinity
