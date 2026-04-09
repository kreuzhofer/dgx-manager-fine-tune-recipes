"""
evaluate.py — Compare base vs fine-tuned model on SQL generation.

Loads both models one at a time, runs them on test examples, and produces:
  1. Accuracy numbers (printed to terminal)
  2. Side-by-side examples showing where the fine-tuned model improved
  3. A bar chart saved as PNG
  4. Detailed results saved as JSON + Markdown

Usage:
  python evaluate.py --base-model google/gemma-4-e2b \
                     --tuned-model /workspace/outputs/{jobId}/merged \
                     --num-examples 100
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.patches import apply_all
apply_all()

import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    """Ask the model to generate SQL for a given schema + question."""
    messages = [
        {
            "role": "system",
            "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else.",
        },
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
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
    """Normalize SQL for comparison."""
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate_model(model_path, test_data, ground_truth, label):
    """Load a model, run it on test data, return predictions."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--dataset", default="b-mc2/sql-create-context")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--results-dir", default="/workspace/outputs/eval")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="train")
    test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    num = min(args.num_examples, len(test_split))
    test_data = test_split.select(range(num))
    ground_truth = [ex["answer"] for ex in test_data]
    print(f"Evaluating on {num} test examples")

    # Evaluate both models
    base_preds = evaluate_model(args.base_model, test_data, ground_truth, "Base Model")
    tuned_preds = evaluate_model(args.tuned_model, test_data, ground_truth, "Fine-tuned Model")

    # Compute accuracy
    base_correct = sum(normalize_sql(p) == normalize_sql(g) for p, g in zip(base_preds, ground_truth))
    tuned_correct = sum(normalize_sql(p) == normalize_sql(g) for p, g in zip(tuned_preds, ground_truth))
    base_acc = base_correct / num * 100
    tuned_acc = tuned_correct / num * 100

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS ({num} test examples)")
    print(f"{'='*60}")
    print(f"  Base model accuracy:       {base_acc:5.1f}%  ({base_correct}/{num})")
    print(f"  Fine-tuned model accuracy: {tuned_acc:5.1f}%  ({tuned_correct}/{num})")
    print(f"  Improvement:               {tuned_acc - base_acc:+5.1f}%")
    print(f"{'='*60}")

    # Show examples where fine-tuning helped
    print(f"\nExamples where fine-tuning FIXED the output:\n")
    shown = 0
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

    # Generate chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Base", "Fine-tuned"], [base_acc, tuned_acc], color=["#2196F3", "#4CAF50"], width=0.5)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title(f"SQL Generation: Base vs Fine-tuned ({num} examples)")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, [base_acc, tuned_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{acc:.1f}%",
                ha="center", fontweight="bold", fontsize=14)
    plt.tight_layout()
    chart_path = os.path.join(args.results_dir, "accuracy_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    # Save JSON
    results_path = os.path.join(args.results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "tuned_model": args.tuned_model,
            "num_examples": num,
            "base_accuracy_pct": round(base_acc, 2),
            "tuned_accuracy_pct": round(tuned_acc, 2),
            "improvement_pct": round(tuned_acc - base_acc, 2),
            "examples": [{
                "question": test_data[i]["question"],
                "schema": test_data[i]["context"],
                "ground_truth": ground_truth[i],
                "base_prediction": base_preds[i],
                "tuned_prediction": tuned_preds[i],
                "base_correct": normalize_sql(base_preds[i]) == normalize_sql(ground_truth[i]),
                "tuned_correct": normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i]),
            } for i in range(num)],
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
