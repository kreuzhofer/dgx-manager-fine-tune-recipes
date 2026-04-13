"""Dataset loading and formatting for training.

Supports multiple dataset formats:
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
- OpenAI:   {"conversations": [{"role": "user", "content": "..."}, ...]}
- QA:       {"question": "...", "answer": "..."} with optional "context"
- Instruct: {"instruction": "...", "output": "..."} with optional "input"
"""

import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


class Gemma4DataCollator(DataCollatorForLanguageModeling):
    """Data collator that passes through mm_token_type_ids and token_type_ids.

    The default collator drops these fields which Gemma 4 requires during training.
    """

    def __call__(self, features, return_tensors=None):
        # Extract extra fields before the parent collator drops them
        mm_token_type_ids = [f.pop("mm_token_type_ids", None) for f in features]
        token_type_ids = [f.pop("token_type_ids", None) for f in features]

        # Standard collation for input_ids, attention_mask, labels
        batch = super().__call__(features, return_tensors=return_tensors)

        # Add back the extra fields (pad to match batch length)
        if any(t is not None for t in mm_token_type_ids):
            max_len = batch["input_ids"].shape[1]
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
            for i, ids in enumerate(mm_token_type_ids):
                if ids is not None:
                    length = min(len(ids), max_len)
                    batch["mm_token_type_ids"][i, :length] = torch.tensor(ids[:length])

        if any(t is not None for t in token_type_ids):
            max_len = batch["input_ids"].shape[1]
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            for i, ids in enumerate(token_type_ids):
                if ids is not None:
                    length = min(len(ids), max_len)
                    batch["token_type_ids"][i, :length] = torch.tensor(ids[:length])

        return batch


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
    # Do NOT pre-compute labels here — tokenizer.pad() doesn't pad the labels
    # field, causing eval collation to fail when batch items have different
    # lengths. DataCollatorForLanguageModeling(mlm=False) derives labels from
    # input_ids AFTER padding, which is the correct order.
    seq_len = len(tokens["input_ids"])
    tokens["token_type_ids"] = [0] * seq_len
    tokens["mm_token_type_ids"] = [0] * seq_len
    return tokens


def prepare_datasets(path_or_id, tokenizer, max_seq_length, eval_fraction=0.1, seed=42, world_rank=0):
    """Load, tokenize, and split a dataset into train/eval sets.

    Returns (train_dataset, eval_dataset). eval_dataset is None if eval_fraction <= 0.
    """
    raw = load_training_dataset(path_or_id)

    if world_rank == 0:
        print(f"Dataset columns: {raw.column_names}, {len(raw)} examples", flush=True)

    # load_from_cache_file=False: HF datasets fingerprint hashing misses changes
    # behind a `lambda`, so edits to format_example don't invalidate the cache.
    # We pay ~1-2 min to re-tokenize; far cheaper than debugging stale-cache bugs.
    tokenized = raw.map(
        lambda ex: format_example(ex, tokenizer, max_seq_length),
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Tokenizing",
        load_from_cache_file=False,
    )

    if eval_fraction > 0:
        split = tokenized.train_test_split(test_size=eval_fraction, seed=seed)
        if world_rank == 0:
            print(f"Split: {len(split['train'])} train / {len(split['test'])} eval", flush=True)
        return split["train"], split["test"]

    return tokenized, None
