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

# Disable NCCL timeout for large model ZeRO-3 all-gather operations.
# PyTorch hardcodes NCCL timeout to 10min and env vars don't override it
# (known PyTorch bug #124950). ASYNC_ERROR_HANDLING disables the timeout.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=== DGX Manager Fine-Tune ==="
echo "Node: $(hostname)"
echo "Nodes: ${NUM_NODES}, Rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL async error handling: ${TORCH_NCCL_ASYNC_ERROR_HANDLING:-not set}"
echo "Script: ${TRAIN_SCRIPT}"
echo "Args: ${TRAIN_ARGS[*]}"
echo "============================="

# Extract --output_dir from TRAIN_ARGS so we can write per-rank logs there.
# Per-rank files avoid interleaved/garbled output from concurrent NFS writes
# when multiple ranks share one log file. Shell-level tee captures EVERYTHING
# (Python prints, C extensions, PyTorch/DeepSpeed loaders) — Python's
# sys.stdout-based Tee misses native C/C++ output written to fd 1/2 directly.
OUTPUT_DIR=""
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
    if [[ "${TRAIN_ARGS[$i]}" == "--output_dir" ]]; then
        OUTPUT_DIR="${TRAIN_ARGS[$((i+1))]}"
        break
    fi
done
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    if [ "$NODE_RANK" -eq 0 ]; then
        RANK_LOG="$OUTPUT_DIR/train.log"
    else
        RANK_LOG="$OUTPUT_DIR/train-rank${NODE_RANK}.log"
    fi
    echo "[launch] Tee'ing all output to $RANK_LOG"
    # Process substitution: tee writes to file AND inherited stdout (so the
    # agent's docker exec / worker log redirect still see the stream live).
    exec > >(tee -a "$RANK_LOG") 2>&1
fi

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
