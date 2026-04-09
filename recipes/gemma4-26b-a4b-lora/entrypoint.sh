#!/usr/bin/env bash
# Container entrypoint for DGX Manager fine-tune jobs.
# Installs Python dependencies and keeps the container alive.
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q \
    "transformers>=4.51.0" \
    peft \
    datasets \
    "trl>=0.16.0" \
    accelerate \
    deepspeed \
    hf_transfer

echo "=== Ready ==="

# Signal readiness and keep container alive
touch /tmp/.ready
exec sleep infinity
