"""Training output logging with file persistence.

Writes all stdout/stderr to both the console (for the agent pipe)
and a log file in the output directory (for reattach + history).
"""

import os
import sys

from transformers import TrainerCallback


class Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        # Return the real stdout fd for compatibility with tqdm etc.
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("No valid file descriptor")

    def isatty(self):
        return False


def setup_logging(output_dir, filename="train.log"):
    """Install a Tee on stdout/stderr that also writes to a log file.

    Call this at the start of training before any output.
    The log file is created at {output_dir}/{filename}.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Logging to {log_path}", flush=True)


class LogMetricsCallback(TrainerCallback):
    """Print training metrics explicitly to stdout for progress tracking.

    Tqdm overwrites loss output in piped mode, so we print structured
    [TRAIN] and [EVAL] lines that the agent can reliably parse.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            total = state.max_steps if state.max_steps > 0 else "?"
            loss = logs.get("loss", "?")
            lr = logs.get("learning_rate", "?")
            print(f"[TRAIN] step={step}/{total} loss={loss} lr={lr}", flush=True)
        if logs and "eval_loss" in logs:
            print(f"[EVAL] eval_loss={logs['eval_loss']}", flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        print("[EVAL] Running evaluation...", flush=True)
