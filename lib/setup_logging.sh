#!/usr/bin/env bash
# Shell-level logging setup: tee stdout/stderr to a per-rank log file in
# OUTPUT_DIR. Captures EVERYTHING (Python prints, C extensions, PyTorch
# and DeepSpeed native loaders) — Python's sys.stdout-based Tee misses
# output written directly to fd 1/2 by native code.
#
# Per-rank files (train.log for rank 0, train-rank{N}.log for workers)
# avoid garbled interleaving from concurrent NFS writes when ranks share
# a log file.
#
# Usage in a launch script:
#   source "$(cd "$(dirname "$0")/../.." && pwd)/lib/setup_logging.sh"
#   setup_shell_log_tee "$@"
#   exec torchrun ... "$@"   # or exec python ... "$@"

setup_shell_log_tee() {
    local output_dir=""
    local rank="${NODE_RANK:-${RANK:-0}}"
    local prev=""
    for arg in "$@"; do
        if [ "$prev" = "--output_dir" ]; then
            output_dir="$arg"
            break
        fi
        prev="$arg"
    done
    if [ -z "$output_dir" ]; then
        echo "[launch] WARNING: --output_dir not in args; log file not captured"
        return 0
    fi
    mkdir -p "$output_dir"
    local rank_log
    if [ "$rank" = "0" ]; then
        rank_log="$output_dir/train.log"
    else
        rank_log="$output_dir/train-rank${rank}.log"
    fi
    echo "[launch] Tee'ing all output to $rank_log"
    # Process substitution: tee writes to file AND inherited stdout (so
    # the agent's docker exec / worker log redirect still see the stream).
    exec > >(tee -a "$rank_log") 2>&1
}
