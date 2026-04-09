"""Dataset loading and formatting for training.

Supports multiple dataset formats:
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
- OpenAI:   {"conversations": [{"role": "user", "content": "..."}, ...]}
- QA:       {"question": "...", "answer": "..."} with optional "context"
- Instruct: {"instruction": "...", "output": "..."} with optional "input"
"""

from datasets import load_dataset


def load_training_dataset(path_or_id, split="train"):
    """Load a dataset from a local file or HuggingFace ID."""
    if path_or_id.endswith(".jsonl") or path_or_id.endswith(".json"):
        return load_dataset("json", data_files=path_or_id, split=split)
    return load_dataset(path_or_id, split=split)


def format_example(example, tokenizer, max_seq_length):
    """Format a single dataset example into tokenized chat for training."""
    messages = []

    if "conversations" in example:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        for t in example["conversations"]:
            if "from" in t:
                messages.append({"role": role_map.get(t["from"], t["from"]), "content": t["value"]})
            else:
                messages.append({"role": role_map.get(t.get("role", ""), t.get("role", "")), "content": t["content"]})
    elif "question" in example and "answer" in example:
        user_msg = example["question"]
        if example.get("context"):
            user_msg = f"{example['context']}\n\n{example['question']}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["answer"]},
        ]
    elif "instruction" in example and "output" in example:
        user_msg = example["instruction"]
        if example.get("input"):
            user_msg = f"{example['instruction']}\n\n{example['input']}"
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]},
        ]
    else:
        raise ValueError(f"Unknown dataset format. Columns: {list(example.keys())}")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(
        text, truncation=True, max_length=max_seq_length, return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def prepare_datasets(path_or_id, tokenizer, max_seq_length, eval_fraction=0.1, seed=42, world_rank=0):
    """Load, tokenize, and split a dataset into train/eval sets.

    Returns (train_dataset, eval_dataset). eval_dataset is None if eval_fraction <= 0.
    """
    raw = load_training_dataset(path_or_id)

    if world_rank == 0:
        print(f"Dataset columns: {raw.column_names}, {len(raw)} examples", flush=True)

    tokenized = raw.map(
        lambda ex: format_example(ex, tokenizer, max_seq_length),
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    if eval_fraction > 0:
        split = tokenized.train_test_split(test_size=eval_fraction, seed=seed)
        if world_rank == 0:
            print(f"Split: {len(split['train'])} train / {len(split['test'])} eval", flush=True)
        return split["train"], split["test"]

    return tokenized, None
