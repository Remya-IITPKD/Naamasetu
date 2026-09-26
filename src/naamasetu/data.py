"""JSONL I/O and HuggingFace dataset helpers for token-classification NER.

Record format (one JSON object per line):
    {"id": "mr_17", "tokens": ["...", ...], "ner_tags": ["B-PER", "O", ...]}
`ner_tags` may also be Naamapadam integer ids.
"""

import json
import os

from .labels import TAG2ID, decode_tags


def read_jsonl(path, max_lines=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
                if max_lines and len(out) >= max_lines:
                    break
    return out


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_ner_jsonl(path, verbose=True):
    """Return (list_of_token_lists, list_of_tag_lists); drops malformed rows."""
    toks, tags = [], []
    for r in read_jsonl(path):
        t, g = r.get("tokens"), r.get("ner_tags")
        if not t or not g or len(t) != len(g):
            continue
        g = decode_tags(g)
        if any(x not in TAG2ID for x in g):
            continue
        toks.append(t)
        tags.append(g)
    if verbose:
        print(f"  {len(toks):>8,}  {os.path.basename(path)}")
    return toks, tags


def tokenize_and_align(examples, tokenizer, max_length=256):
    """Label the first sub-token of each word; mask the rest with -100."""
    enc = tokenizer(examples["tokens"], truncation=True,
                    is_split_into_words=True, max_length=max_length)
    labels = []
    for i, lab in enumerate(examples["ner_tags"]):
        prev, ids = None, []
        for wi in enc.word_ids(batch_index=i):
            if wi is None or wi == prev:
                ids.append(-100)
            else:
                ids.append(TAG2ID[lab[wi]])
            prev = wi
        labels.append(ids)
    enc["labels"] = labels
    return enc


def make_dataset(toks, tags, tokenizer, max_length=256):
    from datasets import Dataset
    return Dataset.from_dict({"tokens": toks, "ner_tags": tags}).map(
        lambda x: tokenize_and_align(x, tokenizer, max_length),
        batched=True, remove_columns=["tokens", "ner_tags"])
