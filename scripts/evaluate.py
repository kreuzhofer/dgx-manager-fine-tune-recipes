"""
evaluate.py — Compare base vs fine-tuned model on SQL generation.

Two execution modes:
  - HF mode: load model weights via transformers (--*-model). Slow, needs
    GPU, holds the whole model in VRAM. Useful when no vLLM endpoint exists.
  - HTTP mode: hit a deployed vLLM /v1/chat/completions (--*-endpoint +
    --*-served-name). Fast, supports --concurrency, does not re-load weights.

Either side is independently optional. Pass only --base-* to get a baseline
number in isolation (useful before training starts).

Usage (HF, base vs tuned):
  python evaluate.py --base-model google/gemma-4-e2b \
                     --tuned-model /workspace/outputs/{jobId}/merged \
                     --num-examples 100

Usage (HTTP, base only — baseline):
  python evaluate.py \
      --base-endpoint http://192.168.44.37:8000 \
      --base-served-name Qwen/Qwen3.6-35B-A3B \
      --num-examples 100 --concurrency 4 \
      --results-dir /mnt/tank/results/sql-baseline-qwen3.6
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Heavy HF deps are only needed for HF mode. Defer the import so HTTP-only
# runs (the common case post-fine-tune) don't require a CUDA host.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


def _ensure_hf_imports():
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is not None:
        return
    from lib.patches import apply_all
    apply_all()
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AMCLM, AutoTokenizer as _AT
    torch = _torch
    AutoModelForCausalLM = _AMCLM
    AutoTokenizer = _AT


def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    """Ask the model to generate SQL for a given schema + question.

    Uses the same format as lib/dataset.py format_example() for QA datasets
    so the prompt matches what the model was trained on.
    """
    messages = [
        {"role": "user", "content": f"{schema}\n\n{question}"},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def normalize_sql(sql):
    """Normalize SQL for comparison.

    Handles instruction-tuned models that wrap SQL in markdown code blocks
    with natural language explanation (common with -it models like 26B-A4B-it).
    """
    import re
    s = sql.strip()
    # Extract SQL from markdown code blocks: ```sql\n...\n``` or ```\n...\n```
    code_block = re.search(r"```(?:sql)?\s*\n?(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if code_block:
        s = code_block.group(1).strip()
    # Remove chat template artifacts
    for tag in ["<end_of_turn>", "<start_of_turn>", "model", "user"]:
        s = s.split(tag)[0]
    s = s.strip().rstrip(";")
    # Normalize quotes (single → double)
    s = s.replace("'", '"')
    return " ".join(s.lower().split())


def evaluate_model(model_path, test_data, ground_truth, label):
    """Load a model, run it on test data, return predictions."""
    _ensure_hf_imports()
    print(f"\n{'='*60}")
    print(f"Evaluating (HF): {label}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma 4 chat template fallback
    if not getattr(tokenizer, 'chat_template', None):
        from lib.tokenizer import GEMMA_CHAT_TEMPLATE
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    predictions = []
    for i, example in enumerate(test_data):
        sql = generate_sql(model, tokenizer, example["context"], example["question"])
        predictions.append(sql)
        is_correct = normalize_sql(sql) == normalize_sql(ground_truth[i])
        if (i + 1) % 10 == 0 or i == 0:
            mark = "OK" if is_correct else "MISS"
            print(f"  [{i+1}/{len(test_data)}] [{mark}] Q: {example['question'][:60]}...")
            print(f"           Predicted: {sql[:80]}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return predictions


def generate_sql_http(endpoint, served_name, schema, question, max_tokens=256,
                     api_key=None, timeout=300.0):
    """POST to a vLLM /v1/chat/completions endpoint. Returns SQL string."""
    import requests
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": f"{schema}\n\n{question}"}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def evaluate_model_http(endpoint, served_name, test_data, ground_truth, label,
                        max_tokens=256, concurrency=1, api_key=None):
    """Evaluate a deployed vLLM endpoint. Concurrency > 1 fires parallel
    requests via a thread pool — vLLM batches server-side."""
    print(f"\n{'='*60}")
    print(f"Evaluating (HTTP): {label}")
    print(f"Endpoint: {endpoint}  served_name={served_name}  c={concurrency}")
    print(f"{'='*60}")

    n = len(test_data)
    predictions = [""] * n
    errors = [None] * n

    def _one(i):
        ex = test_data[i]
        try:
            predictions[i] = generate_sql_http(
                endpoint, served_name, ex["context"], ex["question"],
                max_tokens=max_tokens, api_key=api_key)
        except Exception as e:
            errors[i] = f"{type(e).__name__}: {e}"
        return i

    completed = 0
    if concurrency <= 1:
        for i in range(n):
            _one(i)
            completed += 1
            if completed % 10 == 0 or completed == 1:
                ok = errors[i] is None and normalize_sql(predictions[i]) == normalize_sql(ground_truth[i])
                mark = "OK" if ok else ("ERR" if errors[i] else "MISS")
                print(f"  [{completed}/{n}] [{mark}] {test_data[i]['question'][:60]}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one, i) for i in range(n)]
            for fut in as_completed(futures):
                i = fut.result()
                completed += 1
                if completed % 10 == 0 or completed == 1:
                    ok = errors[i] is None and normalize_sql(predictions[i]) == normalize_sql(ground_truth[i])
                    mark = "OK" if ok else ("ERR" if errors[i] else "MISS")
                    print(f"  [{completed}/{n}] [{mark}] {test_data[i]['question'][:60]}")

    n_err = sum(1 for e in errors if e)
    if n_err:
        print(f"  WARNING: {n_err}/{n} requests failed (treated as empty predictions)")

    return predictions


def _evaluate_side(args, side, test_data, ground_truth):
    """Run one side (base or tuned). Returns predictions or None if no
    --*-model / --*-endpoint was supplied for this side."""
    model_arg = getattr(args, f"{side}_model", None)
    endpoint_arg = getattr(args, f"{side}_endpoint", None)
    served_name = getattr(args, f"{side}_served_name", None)
    if model_arg and endpoint_arg:
        raise SystemExit(f"--{side}-model and --{side}-endpoint are mutually exclusive")
    label = f"{side.capitalize()} Model"
    if endpoint_arg:
        if not served_name:
            raise SystemExit(
                f"--{side}-endpoint requires --{side}-served-name (the model "
                f"name vLLM is serving, e.g. Qwen/Qwen3.6-35B-A3B)")
        return evaluate_model_http(
            endpoint_arg, served_name, test_data, ground_truth, label,
            max_tokens=args.max_tokens, concurrency=args.concurrency,
            api_key=args.api_key)
    if model_arg:
        return evaluate_model(model_arg, test_data, ground_truth, label)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model on SQL")
    # HF paths OR HTTP endpoints; both sides independently optional.
    parser.add_argument("--base-model", help="HF model id or local path for base (HF mode)")
    parser.add_argument("--tuned-model", help="HF model id or local path for tuned (HF mode)")
    parser.add_argument("--base-endpoint", help="vLLM base URL for base (HTTP mode)")
    parser.add_argument("--tuned-endpoint", help="vLLM base URL for tuned (HTTP mode)")
    parser.add_argument("--base-served-name",
                        help="model name vLLM is serving for base (required with --base-endpoint)")
    parser.add_argument("--tuned-served-name",
                        help="model name vLLM is serving for tuned (required with --tuned-endpoint)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="parallel HTTP requests (HTTP mode only)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dataset", default="b-mc2/sql-create-context")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--results-dir", default="/workspace/outputs/eval")
    parser.add_argument("--results-name", default="results.json",
                        help="output JSON filename (chart uses the same stem)")
    args = parser.parse_args()

    have_base = bool(args.base_model or args.base_endpoint)
    have_tuned = bool(args.tuned_model or args.tuned_endpoint)
    if not (have_base or have_tuned):
        raise SystemExit("Provide at least one of --base-model/--base-endpoint or "
                         "--tuned-model/--tuned-endpoint")

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="train")
    test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    num = min(args.num_examples, len(test_split))
    test_data = test_split.select(range(num))
    ground_truth = [ex["answer"] for ex in test_data]
    print(f"Evaluating on {num} test examples")

    wall0 = time.time()
    base_preds = _evaluate_side(args, "base", test_data, ground_truth)
    tuned_preds = _evaluate_side(args, "tuned", test_data, ground_truth)
    wall = time.time() - wall0

    def acc(preds):
        if preds is None:
            return None, None
        c = sum(normalize_sql(p) == normalize_sql(g) for p, g in zip(preds, ground_truth))
        return c, c / len(preds) * 100

    base_correct, base_acc = acc(base_preds)
    tuned_correct, tuned_acc = acc(tuned_preds)

    print(f"\n{'='*60}")
    print(f"  RESULTS ({num} test examples, wall {wall:.1f}s)")
    print(f"{'='*60}")
    if base_acc is not None:
        print(f"  Base model accuracy:       {base_acc:5.1f}%  ({base_correct}/{num})")
    if tuned_acc is not None:
        print(f"  Fine-tuned model accuracy: {tuned_acc:5.1f}%  ({tuned_correct}/{num})")
    if base_acc is not None and tuned_acc is not None:
        print(f"  Improvement:               {tuned_acc - base_acc:+5.1f}%")
    print(f"{'='*60}")

    if base_preds and tuned_preds:
        shown = 0
        print(f"\nExamples where fine-tuning FIXED the output:\n")
        for i in range(num):
            base_ok = normalize_sql(base_preds[i]) == normalize_sql(ground_truth[i])
            tuned_ok = normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i])
            if not base_ok and tuned_ok and shown < 5:
                print(f"  Example {i+1}:")
                print(f"    Question:     {test_data[i]['question']}")
                print(f"    Ground truth: {ground_truth[i]}")
                print(f"    Base model:   {base_preds[i]}")
                print(f"    Fine-tuned:   {tuned_preds[i]}")
                print()
                shown += 1

    # Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, values, colors = [], [], []
    if base_acc is not None:
        labels.append("Base"); values.append(base_acc); colors.append("#2196F3")
    if tuned_acc is not None:
        labels.append("Fine-tuned"); values.append(tuned_acc); colors.append("#4CAF50")
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title(f"SQL Generation ({num} examples)")
    ax.set_ylim(0, 100)
    for bar, acc_v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{acc_v:.1f}%",
                ha="center", fontweight="bold", fontsize=14)
    plt.tight_layout()
    stem = os.path.splitext(args.results_name)[0]
    chart_path = os.path.join(args.results_dir, f"{stem}.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    # JSON
    def _side_payload(preds, model_arg, endpoint_arg, served_name, acc_v, correct):
        if preds is None:
            return None
        return {
            "model": model_arg, "endpoint": endpoint_arg, "served_name": served_name,
            "accuracy_pct": round(acc_v, 2) if acc_v is not None else None,
            "correct": correct, "num": len(preds),
            "predictions": preds,
        }

    results_path = os.path.join(args.results_dir, args.results_name)
    with open(results_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "num_examples": num,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "wall_time_s": round(wall, 2),
            "base": _side_payload(base_preds, args.base_model, args.base_endpoint,
                                  args.base_served_name, base_acc, base_correct),
            "tuned": _side_payload(tuned_preds, args.tuned_model, args.tuned_endpoint,
                                   args.tuned_served_name, tuned_acc, tuned_correct),
            "test_examples": [{
                "question": test_data[i]["question"],
                "schema": test_data[i]["context"],
                "ground_truth": ground_truth[i],
            } for i in range(num)],
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
