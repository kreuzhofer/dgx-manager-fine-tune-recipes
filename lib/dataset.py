"""Dataset loading and formatting for training.

Supports multiple dataset formats:
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
- OpenAI legacy: {"conversations": [{"role": "user", "content": "..."}, ...]}
- OpenAI chat completion: {"messages": [{"role": "...", "content": "..."}, ...]}
  (optionally with "tools": [...] for function-calling datasets)
- QA:       {"question": "...", "answer": "..."} with optional "context"
- Instruct: {"instruction": "...", "output": "..."} with optional "input"

Local JSONL files are loaded with strict schema-tolerance: only the
training-relevant fields are retained, so datasets that merge multiple
sources with inconsistent auxiliary columns (e.g. metadata schemas that
gain/lose fields across rows) load without HF's pyarrow cast error.
"""

import json
import torch
from datasets import Dataset, load_dataset
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


# Top-level fields format_example knows how to consume. Anything outside
# this set (auxiliary metadata, scores, IDs, etc.) is dropped at load time
# for local JSONL files so HF's strict schema cast never sees them.
_TRAINING_FIELDS = frozenset({
    "conversations",       # ShareGPT / legacy "OpenAI" with conversations array
    "messages",            # OpenAI chat completion format
    "tools",               # OpenAI function-calling tool definitions
    "system",              # optional standalone system prompt
    "question", "answer", "context",
    "instruction", "input", "output",
})


def _load_jsonl_training_only(path):
    """Stream-load a local JSONL, retain only training-relevant fields.

    Works around HF datasets' strict pyarrow schema cast: when a JSONL
    merges two sources whose auxiliary columns (e.g. `metadata` with
    different sub-keys per source) differ, `load_dataset("json", ...)`
    infers the schema from the first batch then fails to cast later
    rows. By keeping only fields the trainer actually consumes, we
    bypass schema inference entirely.

    Raises if a row has none of the expected training fields, since
    that would mean the file is structurally wrong rather than just
    schema-noisy."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} of {path}: {e}") from e
            cleaned = {k: v for k, v in row.items() if k in _TRAINING_FIELDS}
            if not cleaned:
                raise ValueError(
                    f"Row {line_no} of {path} has no recognized training fields. "
                    f"Got top-level keys: {sorted(row.keys())}. "
                    f"Expected at least one of: {sorted(_TRAINING_FIELDS)}"
                )
            rows.append(cleaned)
    return Dataset.from_list(rows)


def load_training_dataset(path_or_id, split="train"):
    """Load a dataset from a local file or HuggingFace ID.

    Local .jsonl/.json files use a schema-tolerant streaming loader
    that only retains training-relevant fields (see
    `_load_jsonl_training_only`). HuggingFace hub IDs go through the
    standard `load_dataset` path."""
    if path_or_id.endswith(".jsonl") or path_or_id.endswith(".json"):
        return _load_jsonl_training_only(path_or_id)
    return load_dataset(path_or_id, split=split)


def format_example(example, tokenizer, max_seq_length):
    """Format a single dataset example into tokenized chat for training."""
    messages = []
    tools = None

    if "messages" in example:
        # OpenAI chat completion format. Pass through directly; let the
        # tokenizer's chat template render. Carry `tools` for
        # function-calling datasets — modern chat templates accept it
        # as a kwarg and silently no-op when unused.
        messages = example["messages"]
        tools = example.get("tools")
    elif "conversations" in example:
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

    chat_template_kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools is not None:
        chat_template_kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
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
    # keep_in_memory=True: skip writing to the NFS-backed cache. Without this,
    # multi-rank training has every rank simultaneously writing the same
    # `.arrow` files; the resulting size races mean a rank later mmaps a file
    # shorter than its on-disk header advertises and dies with SIGBUS deep
    # inside pyarrow. The dataset is small enough to fit in RAM (~78K
    # examples × 256 tokens × ~10B per token = ~200 MB), so paying the
    # tokenization cost in-memory is cheap insurance.
    tokenized = raw.map(
        lambda ex: format_example(ex, tokenizer, max_seq_length),
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Tokenizing",
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    if eval_fraction > 0:
        split = tokenized.train_test_split(test_size=eval_fraction, seed=seed)
        if world_rank == 0:
            print(f"Split: {len(split['train'])} train / {len(split['test'])} eval", flush=True)
        return split["train"], split["test"]

    return tokenized, None
