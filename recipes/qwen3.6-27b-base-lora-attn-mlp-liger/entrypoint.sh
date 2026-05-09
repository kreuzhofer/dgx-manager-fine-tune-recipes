#!/usr/bin/env bash
# Container entrypoint for multi-node DeepSpeed training.
# Installs dependencies, sets up SSH for inter-node communication.
set -euo pipefail

echo "=== Setting up SSH ==="
apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null 2>&1
mkdir -p /var/run/sshd /root/.ssh

# Copy SSH keys from read-only mount (host keys mapped to /tmp/.ssh)
cp /tmp/.ssh/* /root/.ssh/ 2>/dev/null || true
chmod 700 /root/.ssh
chmod 600 /root/.ssh/id_* 2>/dev/null || true
chmod 644 /root/.ssh/*.pub 2>/dev/null || true

# SSH on port 2233 (avoid conflicts with host sshd)
cat > /etc/ssh/sshd_config.d/training.conf << 'EOF'
Port 2233
PermitRootLogin yes
StrictModes no
EOF
echo "StrictHostKeyChecking no" >> /root/.ssh/config

echo "=== Installing dependencies ==="
# peft >= 0.17.0 needed for target_parameters to reach fused-expert MoE
# tensors. Older PEFT silently skips them, leaving only ~30M attention LoRA.
pip install -q \
    "transformers>=4.51.0" \
    "peft>=0.17.0" \
    datasets \
    "trl>=0.16.0" \
    accelerate \
    deepspeed \
    hf_transfer \
    "liger-kernel>=0.5.0"
echo "PEFT version: $(python -c 'import peft; print(peft.__version__)')"
echo "Liger import probe: $(python -c 'import liger_kernel; from liger_kernel.transformers import monkey_patch; print(\"ok (top-level + monkey_patch import)\")' 2>&1 | tail -1)"

echo "=== Ready ==="

# Start SSH daemon and signal readiness
/usr/sbin/sshd
touch /tmp/.ready
exec sleep infinity
