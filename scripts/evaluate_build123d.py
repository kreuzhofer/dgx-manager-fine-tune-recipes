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

Usage:
  python evaluate_build123d.py \\
      --base-model google/gemma-4-26B-A4B \\
      --tuned-model /workspace/outputs/{jobId}/merged \\
      --dataset /workspace/datasets/{id}/training-data.jsonl \\
      --num-examples 50 \\
      --results-dir /workspace/outputs/{jobId}/build123d-eval
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.patches import apply_all
apply_all()

import argparse
import ast
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def generate(model, tokenizer, system: str, user: str,
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


def evaluate_model(model_path: str, examples: list[dict],
                   whitelist: set[str], label: str) -> tuple[list[str], list[dict]]:
    print(f"\n{'=' * 60}\nEvaluating: {label}\nModel path: {model_path}\n{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        from lib.tokenizer import GEMMA_CHAT_TEMPLATE
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto")

    predictions, scores = [], []
    for i, ex in enumerate(examples):
        pred = generate(model, tokenizer, ex["system"], ex["human"])
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
    return predictions, scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned on build123d generation")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Path to JSONL training/eval file (ShareGPT 3-turn format)")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling test split")
    parser.add_argument("--test-fraction", type=float, default=0.05,
                        help="held-out fraction (matches lib/dataset.py default)")
    parser.add_argument("--results-dir", default="/workspace/outputs/build123d-eval")
    args = parser.parse_args()

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
    train_indices = indices[n_test:]

    test_examples = [all_examples[i] for i in test_indices][:args.num_examples]
    print(f"Test set: {len(test_examples)} examples (held-out via seed={args.seed}, "
          f"test_fraction={args.test_fraction})")

    # Build whitelist from ALL reference codes (train + test) — symbols used
    # anywhere in the dataset are valid. Anything else from the model is a hallucination.
    all_ref_codes = [all_examples[i]["reference_code"] for i in range(len(all_examples))]
    whitelist = build_api_whitelist(all_ref_codes)
    print(f"API whitelist size: {len(whitelist)} CamelCase symbols (from references + safe set)")

    base_preds, base_scores = evaluate_model(
        args.base_model, test_examples, whitelist, "Base Model")
    tuned_preds, tuned_scores = evaluate_model(
        args.tuned_model, test_examples, whitelist, "Fine-tuned Model")

    base_agg = aggregate_scores(base_scores)
    tuned_agg = aggregate_scores(tuned_scores)

    print(f"\n{'=' * 60}\n  RESULTS ({len(test_examples)} test examples)\n{'=' * 60}")
    print(f"  {'Metric':<22}  {'Base':>8}  {'Tuned':>8}  {'Δ':>8}")
    print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for k in ["has_python_block", "parses", "defines_root_part", "no_forbidden",
              "api_in_whitelist", "length_sanity"]:
        b, t = base_agg[k], tuned_agg[k]
        print(f"  {k:<22}  {b:>7.1f}%  {t:>7.1f}%  {t - b:>+7.1f}%")
    print(f"  {'api_coverage_mean':<22}  {base_agg['api_coverage_mean']:>8.3f}  "
          f"{tuned_agg['api_coverage_mean']:>8.3f}  "
          f"{tuned_agg['api_coverage_mean'] - base_agg['api_coverage_mean']:>+8.3f}")
    print(f"  {'composite_mean':<22}  {base_agg['composite_mean']:>8.3f}  "
          f"{tuned_agg['composite_mean']:>8.3f}  "
          f"{tuned_agg['composite_mean'] - base_agg['composite_mean']:>+8.3f}")
    print("=" * 60)

    # Show 2 examples where base failed and tuned succeeded
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

    # Chart
    metric_keys = ["has_python_block", "parses", "defines_root_part",
                   "no_forbidden", "api_in_whitelist", "length_sanity"]
    base_vals = [base_agg[k] for k in metric_keys]
    tuned_vals = [tuned_agg[k] for k in metric_keys]
    x = range(len(metric_keys))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base",
           color="#2196F3")
    ax.bar([i + width / 2 for i in x], tuned_vals, width, label="Fine-tuned",
           color="#4CAF50")
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace("_", "\n") for k in metric_keys], fontsize=9)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"build123d Generation: Base vs Fine-tuned ({len(test_examples)} examples)")
    ax.legend()
    for i, (b, t) in enumerate(zip(base_vals, tuned_vals)):
        ax.text(i - width / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, t + 1.5, f"{t:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    chart_path = Path(args.results_dir) / "build123d_comparison.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    # Per-example results
    results_path = Path(args.results_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "tuned_model": args.tuned_model,
            "dataset": args.dataset,
            "num_examples": len(test_examples),
            "whitelist_size": len(whitelist),
            "base_aggregate": base_agg,
            "tuned_aggregate": tuned_agg,
            "examples": [{
                "human": ex["human"],
                "reference_code": ex["reference_code"],
                "base_prediction": base_preds[i],
                "base_scores": base_scores[i],
                "tuned_prediction": tuned_preds[i],
                "tuned_scores": tuned_scores[i],
            } for i, ex in enumerate(test_examples)],
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
