#!/usr/bin/env bash
# Launch DeepSpeed ZeRO-3 training — supports single-node and multi-node.
#
# Single-node: launch.sh --model_name ... --dataset ...
# Multi-node:  launch.sh --hostfile /tmp/hostfile.txt --model_name ... --dataset ...
#
# For multi-node, this script must be run on EACH node independently.
# Nodes discover their rank by matching local IPs against the hostfile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
DS_CONFIG="${SCRIPT_DIR}/ds_config.json"

# Extract --hostfile if provided, pass remaining args to train.py
HOSTFILE=""
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "$HOSTFILE" ] && [ -f "$HOSTFILE" ]; then
    # Multi-node mode
    MASTER_ADDR="$(head -1 "$HOSTFILE" | awk '{print $1}')"
    NUM_NODES=$(grep -c . "$HOSTFILE")

    # Find our rank by matching local IPs against hostfile
    ALL_IPS=$(hostname -I 2>/dev/null || ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    NODE_RANK=-1
    RANK_IDX=0
    while IFS= read -r line; do
        HOST_IP=$(echo "$line" | awk '{print $1}')
        for ip in $ALL_IPS; do
            if [ "$HOST_IP" = "$ip" ]; then
                NODE_RANK=$RANK_IDX
                break 2
            fi
        done
        RANK_IDX=$((RANK_IDX + 1))
    done < "$HOSTFILE"

    if [ "$NODE_RANK" -lt 0 ]; then
        echo "ERROR: None of this node's IPs match the hostfile."
        echo "Node IPs: $ALL_IPS"
        cat "$HOSTFILE"
        exit 1
    fi
else
    # Single-node mode
    NUM_NODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
fi

# Find available master port
MASTER_PORT=$(python3 -c "
import socket
port = 9901
while port < 10000:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', port))
        s.close()
        print(port)
        break
    except OSError:
        port += 1
else:
    print(9901)
")

# Flush page cache
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Increase NCCL timeout for large model ZeRO-3 all-gather operations
# Default 10min is too short for 26B model on DGX Spark
export NCCL_TIMEOUT=14400  # 4 hours in seconds
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export TORCH_DIST_INIT_BARRIER=1

echo "=== DGX Manager Fine-Tune ==="
echo "Node: $(hostname)"
echo "Nodes: ${NUM_NODES}, Rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL timeout: ${NCCL_TIMEOUT}s"
echo "Script: ${TRAIN_SCRIPT}"
echo "Args: ${TRAIN_ARGS[*]}"
echo "============================="

exec torchrun \
    --nnodes="$NUM_NODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node=1 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=static \
    "$TRAIN_SCRIPT" \
    --ds_config "$DS_CONFIG" \
    "${TRAIN_ARGS[@]}"
