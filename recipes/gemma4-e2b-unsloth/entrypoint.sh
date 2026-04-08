#!/usr/bin/env bash
# Container entrypoint for Unsloth fine-tune jobs.
# The dgx-spark-unsloth image already has all dependencies installed.
set -euo pipefail

echo "=== Unsloth Fine-Tune Container Ready ==="
echo "Unsloth and dependencies pre-installed in image."

# Signal readiness and keep container alive
touch /tmp/.ready
exec sleep infinity
