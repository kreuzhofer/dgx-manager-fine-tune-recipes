"""evaluate_build123d.py — Compare base vs fine-tuned models on build123d code generation.

This is a CHEAP proxy eval that does NOT execute the generated geometry. The
ground-truth geometry pipeline lives in another project; this script gives a
fast directional signal during/after training.

Metrics (per generation, then aggregated):
  1. has_python_block   — output contains a fenced ```python ... ``` block
  2. parses             — extracted code parses with ast.parse()
  3. defines_root_part  — code has a top-level assignment to `root_part`
  4. no_forbidden       — no banned imports (sys, matplotlib, ocp_vscode, os, ...)
  5. api_in_whitelist   — every CamelCase call is a known build123d/bd_warehouse symbol
  6. api_coverage       — Jaccard of build123d symbols used vs reference
  7. length_sanity      — 0.5x to 2.0x reference length

The whitelist is built dynamically from the union of CamelCase symbols used
in ALL reference solutions in the dataset, plus a small static list of
known-good helpers (math, range, etc.). This means the whitelist auto-grows
with the dataset.

Two execution modes:
  - HF mode: load model weights with transformers (slow, blocks GPU). Pass
    --base-model / --tuned-model as HF paths (or local merged-model dirs).
  - HTTP mode: hit a deployed vLLM OpenAI-compatible endpoint. Pass
    --base-endpoint / --tuned-endpoint. Optional --*-served-name to override
    the model name in the request payload (default: HF id served by vLLM).
    HTTP mode also unlocks parallelism (--concurrency), so a 50-example run
    over a deployed endpoint takes seconds instead of an hour.

Either side (base or tuned) is optional — if you pass only one, the script
runs single-model mode and emits a single-bar chart. Useful for scoring a
deployed base model in isolation.

Usage (HF mode, base vs tuned):
  python evaluate_build123d.py \\
      --base-model google/gemma-4-26B-A4B \\
      --tuned-model /workspace/outputs/{jobId}/merged \\
      --dataset /workspace/datasets/{id}/training-data.jsonl \\
      --num-examples 50 \\
      --results-dir /workspace/outputs/{jobId}/build123d-eval

Usage (HTTP mode, base only):
  python evaluate_build123d.py \\
      --base-endpoint http://192.168.44.36:8000 \\
      --base-served-name Qwen/Qwen3.6-35B-A3B-FP8 \\
      --dataset /workspace/datasets/{id}/training-data.jsonl \\
      --num-examples 50 --concurrency 8 \\
      --results-dir /workspace/results/qwen3.6-fp8-solo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# transformers + torch are heavy and only needed for HF mode. Defer the import
# so that HTTP-only runs (the common path now) don't eat a 30s import + GPU
# probe and don't fail on machines without CUDA.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
apply_all = None


def _ensure_hf_imports():
    """Lazy-import transformers/torch + the local lib.patches helpers. Called
    only when HF mode is actually selected."""
    global torch, AutoModelForCausalLM, AutoTokenizer, apply_all
    if torch is not None:
        return
    from lib.patches import apply_all as _apply_all
    _apply_all()
    apply_all = _apply_all
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AMCLM, AutoTokenizer as _AT
    torch = _torch
    AutoModelForCausalLM = _AMCLM
    AutoTokenizer = _AT


FORBIDDEN_IMPORTS = {
    "sys", "os", "subprocess", "shutil", "matplotlib", "ocp_vscode",
    "tkinter", "pyqt5", "pyqt6", "pyside2", "pyside6",
}

# Always-OK symbols that aren't build123d primitives but are legitimate
# Python / stdlib / numeric building blocks the model may use.
GENERIC_SAFE_CALLS = {
    "range", "len", "min", "max", "abs", "int", "float", "str", "list",
    "tuple", "dict", "set", "sum", "enumerate", "zip", "round",
    "Vector", "Vertex", "Axis", "Plane",  # build123d geometric primitives
}

CAMEL_CALL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\s*\(")
PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
ANY_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)\n```", re.DOTALL)


def extract_code(text: str) -> tuple[str | None, bool]:
    """Return (code, has_python_block). If no python block, fall back to any
    fenced block so we can still parse-check, but mark has_python_block=False."""
    m = PYTHON_BLOCK_RE.search(text)
    if m:
        return m.group(1), True
    m = ANY_BLOCK_RE.search(text)
    if m:
        return m.group(1), False
    return None, False


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def defines_root_part(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "root_part":
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "root_part":
                return True
    return False


def has_forbidden_imports(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False  # can't parse means we can't say it's forbidden
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return True
    return False


def extract_camel_calls(code: str) -> set[str]:
    """Find CamelCase function/class calls in source — proxy for API usage."""
    return set(CAMEL_CALL_RE.findall(code))


def build_api_whitelist(reference_codes: list[str]) -> set[str]:
    """Whitelist = union of CamelCase calls across all reference solutions
    + GENERIC_SAFE_CALLS. Anything outside this set is treated as a
    hallucinated symbol."""
    wl = set(GENERIC_SAFE_CALLS)
    for code in reference_codes:
        wl |= extract_camel_calls(code)
    return wl


def api_in_whitelist(code: str, whitelist: set[str]) -> tuple[bool, list[str]]:
    """Return (all_in_whitelist, list_of_unknown_symbols)."""
    calls = extract_camel_calls(code)
    unknown = sorted(calls - whitelist)
    return (len(unknown) == 0, unknown)


def api_coverage(generated_code: str, reference_code: str,
                 whitelist: set[str]) -> float:
    """Jaccard similarity of build123d symbols (intersected with whitelist)
    used in generated vs reference. Returns 0.0 if reference has no API calls."""
    gen_calls = extract_camel_calls(generated_code) & whitelist
    ref_calls = extract_camel_calls(reference_code) & whitelist
    if not ref_calls:
        return 1.0 if not gen_calls else 0.0
    union = gen_calls | ref_calls
    return len(gen_calls & ref_calls) / len(union) if union else 0.0


def length_sanity(generated: str, reference: str) -> bool:
    if not reference:
        return False
    ratio = len(generated) / len(reference)
    return 0.5 <= ratio <= 2.0


def score_generation(prediction: str, reference_code: str,
                     whitelist: set[str]) -> dict:
    code, has_py = extract_code(prediction)
    if code is None:
        return {
            "has_python_block": False, "parses": False,
            "defines_root_part": False, "no_forbidden": True,
            "api_in_whitelist": False, "unknown_symbols": [],
            "api_coverage": 0.0,
            "length_sanity": length_sanity(prediction, reference_code),
            "composite": 0.0,
        }
    parses_ok = parses(code)
    root = defines_root_part(code)
    forbidden = has_forbidden_imports(code)
    in_wl, unknown = api_in_whitelist(code, whitelist)
    coverage = api_coverage(code, reference_code, whitelist)
    length_ok = length_sanity(code, reference_code)

    hard = [has_py, parses_ok, root, not forbidden, in_wl]
    soft = [coverage, 1.0 if length_ok else 0.0]
    composite = (sum(hard) / len(hard)) * 0.7 + (sum(soft) / len(soft)) * 0.3

    return {
        "has_python_block": has_py,
        "parses": parses_ok,
        "defines_root_part": root,
        "no_forbidden": not forbidden,
        "api_in_whitelist": in_wl,
        "unknown_symbols": unknown,
        "api_coverage": round(coverage, 3),
        "length_sanity": length_ok,
        "composite": round(composite, 3),
    }


def aggregate_scores(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    boolean_keys = ["has_python_block", "parses", "defines_root_part",
                    "no_forbidden", "api_in_whitelist", "length_sanity"]
    out = {k: round(sum(s[k] for s in scores) / n * 100, 1) for k in boolean_keys}
    out["api_coverage_mean"] = round(sum(s["api_coverage"] for s in scores) / n, 3)
    out["composite_mean"] = round(sum(s["composite"] for s in scores) / n, 3)
    return out


def load_dataset_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            convs = ex.get("conversations", [])
            system = next((c["value"] for c in convs if c["from"] == "system"), "")
            human = next((c["value"] for c in convs if c["from"] == "human"), "")
            gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "")
            ref_code, _ = extract_code(gpt)
            examples.append({
                "system": system, "human": human,
                "gpt_full": gpt, "reference_code": ref_code or "",
            })
    return examples


def generate_hf(model, tokenizer, system: str, user: str,
                max_new_tokens: int = 1024) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_http(endpoint: str, served_name: str, system: str, user: str,
                  max_new_tokens: int = 1024, timeout: float = 600.0,
                  api_key: str | None = None) -> tuple[str, float]:
    """POST to a vLLM /v1/chat/completions endpoint. Returns (text, latency_s)."""
    import requests
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": served_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], dt


def evaluate_model_hf(model_path: str, examples: list[dict],
                      whitelist: set[str], label: str,
                      max_new_tokens: int = 1024,
                      ) -> tuple[list[str], list[dict], list[float]]:
    _ensure_hf_imports()
    print(f"\n{'=' * 60}\nEvaluating (HF): {label}\nModel path: {model_path}\n{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        from lib.tokenizer import GEMMA_CHAT_TEMPLATE
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto")

    predictions, scores, latencies = [], [], []
    for i, ex in enumerate(examples):
        t0 = time.time()
        pred = generate_hf(model, tokenizer, ex["system"], ex["human"], max_new_tokens)
        latencies.append(time.time() - t0)
        predictions.append(pred)
        s = score_generation(pred, ex["reference_code"], whitelist)
        scores.append(s)
        if (i + 1) % 5 == 0 or i == 0:
            mark = "OK" if s["composite"] >= 0.7 else "??"
            print(f"  [{i+1}/{len(examples)}] [{mark}] composite={s['composite']:.2f} "
                  f"parses={s['parses']} root_part={s['defines_root_part']} "
                  f"api_in_wl={s['api_in_whitelist']} cov={s['api_coverage']:.2f}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return predictions, scores, latencies


def evaluate_model_http(endpoint: str, served_name: str, examples: list[dict],
                        whitelist: set[str], label: str,
                        max_new_tokens: int = 1024,
                        concurrency: int = 1,
                        api_key: str | None = None,
                        ) -> tuple[list[str], list[dict], list[float]]:
    """Evaluate a deployed vLLM endpoint. Concurrency > 1 sends parallel
    requests via a thread pool — vLLM batches them server-side, so this is
    typically much faster than serial."""
    print(f"\n{'=' * 60}\nEvaluating (HTTP): {label}\n"
          f"Endpoint: {endpoint}  served_name={served_name}  c={concurrency}\n"
          f"{'=' * 60}")

    n = len(examples)
    predictions: list[str] = [""] * n
    latencies: list[float] = [0.0] * n
    errors: list[str | None] = [None] * n

    def _one(i: int) -> int:
        ex = examples[i]
        try:
            text, dt = generate_http(
                endpoint, served_name, ex["system"], ex["human"],
                max_new_tokens=max_new_tokens, api_key=api_key)
            predictions[i] = text
            latencies[i] = dt
        except Exception as e:
            errors[i] = f"{type(e).__name__}: {e}"
        return i

    completed = 0
    if concurrency <= 1:
        for i in range(n):
            _one(i)
            completed += 1
            if completed % 5 == 0 or completed == 1:
                err = errors[i]
                tag = "OK" if not err else "ERR"
                extra = err if err else f"{latencies[i]:.1f}s"
                print(f"  [{completed}/{n}] [{tag}] {extra}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one, i) for i in range(n)]
            for fut in as_completed(futures):
                i = fut.result()
                completed += 1
                if completed % 5 == 0 or completed == 1:
                    err = errors[i]
                    tag = "OK" if not err else "ERR"
                    extra = err if err else f"{latencies[i]:.1f}s"
                    print(f"  [{completed}/{n}] [{tag}] {extra}")

    n_err = sum(1 for e in errors if e)
    if n_err:
        print(f"  WARNING: {n_err}/{n} requests failed (treated as empty predictions)")

    scores = [score_generation(predictions[i], examples[i]["reference_code"], whitelist)
              for i in range(n)]
    return predictions, scores, latencies


def _evaluate_side(args, side: str, examples: list[dict],
                   whitelist: set[str]
                   ) -> tuple[list[str], list[dict], list[float]] | None:
    """Run one side (base or tuned). Returns None if neither HF nor HTTP args
    were provided for this side."""
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
                f"name vLLM is serving, e.g. Qwen/Qwen3.6-35B-A3B-FP8)")
        return evaluate_model_http(
            endpoint_arg, served_name, examples, whitelist, label,
            max_new_tokens=args.max_tokens, concurrency=args.concurrency,
            api_key=args.api_key)
    if model_arg:
        return evaluate_model_hf(
            model_arg, examples, whitelist, label,
            max_new_tokens=args.max_tokens)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned on build123d generation")
    # HF model paths OR HTTP endpoints; both sides are independently optional
    # so you can score one variant in isolation.
    parser.add_argument("--base-model", help="HF model id or local path for base (HF mode)")
    parser.add_argument("--tuned-model", help="HF model id or local path for tuned (HF mode)")
    parser.add_argument("--base-endpoint", help="vLLM base URL e.g. http://node-ip:8000 (HTTP mode)")
    parser.add_argument("--tuned-endpoint", help="vLLM base URL for tuned model (HTTP mode)")
    parser.add_argument("--base-served-name",
                        help="model name vLLM is serving for base (required with --base-endpoint)")
    parser.add_argument("--tuned-served-name",
                        help="model name vLLM is serving for tuned (required with --tuned-endpoint)")
    parser.add_argument("--api-key", default=None, help="optional bearer token for vLLM endpoint")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="parallel HTTP requests (HTTP mode only; ignored for HF)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dataset", required=True,
                        help="Path to JSONL training/eval file (ShareGPT 3-turn format)")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling test split")
    parser.add_argument("--test-fraction", type=float, default=0.05,
                        help="held-out fraction (matches lib/dataset.py default)")
    parser.add_argument("--results-dir", default="/workspace/outputs/build123d-eval")
    parser.add_argument("--results-name", default="results.json",
                        help="output JSON filename (chart uses the same stem)")
    args = parser.parse_args()

    have_base = bool(args.base_model or args.base_endpoint)
    have_tuned = bool(args.tuned_model or args.tuned_endpoint)
    if not (have_base or have_tuned):
        raise SystemExit("Provide at least one of --base-model/--base-endpoint or "
                         "--tuned-model/--tuned-endpoint")

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    all_examples = load_dataset_jsonl(args.dataset)
    print(f"Loaded {len(all_examples)} examples")

    # Same split semantics as lib/dataset.py: test_size=test_fraction, seed=42
    import random
    rng = random.Random(args.seed)
    indices = list(range(len(all_examples)))
    rng.shuffle(indices)
    n_test = max(1, int(len(all_examples) * args.test_fraction))
    test_indices = indices[:n_test]

    test_examples = [all_examples[i] for i in test_indices][:args.num_examples]
    print(f"Test set: {len(test_examples)} examples (held-out via seed={args.seed}, "
          f"test_fraction={args.test_fraction})")

    # Build whitelist from ALL reference codes (train + test) — symbols used
    # anywhere in the dataset are valid; anything else is a hallucination.
    all_ref_codes = [all_examples[i]["reference_code"] for i in range(len(all_examples))]
    whitelist = build_api_whitelist(all_ref_codes)
    print(f"API whitelist size: {len(whitelist)} CamelCase symbols (from references + safe set)")

    wall0 = time.time()
    base_result = _evaluate_side(args, "base", test_examples, whitelist)
    tuned_result = _evaluate_side(args, "tuned", test_examples, whitelist)
    wall = time.time() - wall0

    metric_keys = ["has_python_block", "parses", "defines_root_part", "no_forbidden",
                   "api_in_whitelist", "length_sanity"]

    print(f"\n{'=' * 60}\n  RESULTS ({len(test_examples)} test examples, wall {wall:.1f}s)\n{'=' * 60}")

    base_agg = aggregate_scores(base_result[1]) if base_result else None
    tuned_agg = aggregate_scores(tuned_result[1]) if tuned_result else None

    if base_agg and tuned_agg:
        print(f"  {'Metric':<22}  {'Base':>8}  {'Tuned':>8}  {'Δ':>8}")
        print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for k in metric_keys:
            b, t = base_agg[k], tuned_agg[k]
            print(f"  {k:<22}  {b:>7.1f}%  {t:>7.1f}%  {t - b:>+7.1f}%")
        for k in ["api_coverage_mean", "composite_mean"]:
            print(f"  {k:<22}  {base_agg[k]:>8.3f}  {tuned_agg[k]:>8.3f}  "
                  f"{tuned_agg[k] - base_agg[k]:>+8.3f}")
    else:
        only = base_agg or tuned_agg
        side_label = "Base" if base_agg else "Tuned"
        print(f"  {'Metric':<22}  {side_label:>8}")
        print(f"  {'-' * 22}  {'-' * 8}")
        for k in metric_keys:
            print(f"  {k:<22}  {only[k]:>7.1f}%")
        for k in ["api_coverage_mean", "composite_mean"]:
            print(f"  {k:<22}  {only[k]:>8.3f}")
    print("=" * 60)

    # Latency summary (useful for HTTP perf signals)
    for label, res in [("base", base_result), ("tuned", tuned_result)]:
        if not res:
            continue
        lats = [x for x in res[2] if x > 0]
        if lats:
            lats_sorted = sorted(lats)
            mean_l = sum(lats) / len(lats)
            p50 = lats_sorted[len(lats_sorted) // 2]
            p95 = lats_sorted[min(len(lats_sorted) - 1, int(0.95 * len(lats_sorted)))]
            print(f"  {label} latency:  mean={mean_l:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s  "
                  f"n={len(lats)}")

    if base_result and tuned_result:
        base_scores = base_result[1]
        tuned_scores = tuned_result[1]
        flips = [i for i in range(len(test_examples))
                 if base_scores[i]["composite"] < 0.5 <= tuned_scores[i]["composite"]]
        if flips:
            print(f"\nExamples where fine-tuning FIXED the output (showing up to 2):\n")
            for i in flips[:2]:
                print(f"  Example {i + 1}: {test_examples[i]['human'][:100]}")
                print(f"    Base composite={base_scores[i]['composite']:.2f}  "
                      f"Tuned composite={tuned_scores[i]['composite']:.2f}")
                print(f"    Base unknown symbols: {base_scores[i]['unknown_symbols'][:5]}")
                print()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(metric_keys))
    if base_agg and tuned_agg:
        width = 0.38
        ax.bar([i - width / 2 for i in x], [base_agg[k] for k in metric_keys], width,
               label="Base", color="#2196F3")
        ax.bar([i + width / 2 for i in x], [tuned_agg[k] for k in metric_keys], width,
               label="Fine-tuned", color="#4CAF50")
        for i, k in enumerate(metric_keys):
            ax.text(i - width / 2, base_agg[k] + 1.5, f"{base_agg[k]:.0f}",
                    ha="center", fontsize=8)
            ax.text(i + width / 2, tuned_agg[k] + 1.5, f"{tuned_agg[k]:.0f}",
                    ha="center", fontsize=8)
        title = f"build123d Generation: Base vs Fine-tuned ({len(test_examples)} examples)"
    else:
        only = base_agg or tuned_agg
        side_label = "Base" if base_agg else "Tuned"
        color = "#2196F3" if base_agg else "#4CAF50"
        ax.bar(list(x), [only[k] for k in metric_keys], 0.6, label=side_label, color=color)
        for i, k in enumerate(metric_keys):
            ax.text(i, only[k] + 1.5, f"{only[k]:.0f}", ha="center", fontsize=8)
        title = f"build123d Generation: {side_label} ({len(test_examples)} examples)"
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace("_", "\n") for k in metric_keys], fontsize=9)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    stem = Path(args.results_name).stem
    chart_path = Path(args.results_dir) / f"{stem}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    def _side_payload(res, model_arg, endpoint_arg, served_name):
        if not res:
            return None
        preds, scores, lats = res
        return {
            "model": model_arg, "endpoint": endpoint_arg, "served_name": served_name,
            "aggregate": aggregate_scores(scores),
            "latencies_s": [round(x, 3) for x in lats],
            "examples": [{"prediction": preds[i], "scores": scores[i]}
                         for i in range(len(scores))],
        }

    results_path = Path(args.results_dir) / args.results_name
    with open(results_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "num_examples": len(test_examples),
            "whitelist_size": len(whitelist),
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "wall_time_s": round(wall, 2),
            "base": _side_payload(base_result, args.base_model, args.base_endpoint,
                                  args.base_served_name),
            "tuned": _side_payload(tuned_result, args.tuned_model, args.tuned_endpoint,
                                   args.tuned_served_name),
            "test_examples": [{"human": ex["human"], "reference_code": ex["reference_code"]}
                              for ex in test_examples],
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
