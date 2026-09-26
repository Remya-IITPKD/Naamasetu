#!/usr/bin/env python3
"""Stage 0 — Download Naamapadam gold splits from the HuggingFace Hub and write
them as JSONL in the format used throughout this repo.

    python scripts/00_prepare_naamapadam.py --langs as gu kn ml mr or pa ta te

Writes data/naamapadam/<lang>/{train,dev,test}.jsonl with string BIO tags.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamsetu.data import write_jsonl  # noqa: E402
from naamsetu.languages import ALL_LANGS  # noqa: E402
from naamsetu.labels import decode_tags  # noqa: E402

SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=ALL_LANGS)
    ap.add_argument("--out-dir", default="data/naamapadam")
    ap.add_argument("--hf-id", default="ai4bharat/naamapadam")
    args = ap.parse_args()

    from datasets import load_dataset
    for lang in args.langs:
        ds = load_dataset(args.hf_id, lang)
        names = ds["train"].features["ner_tags"].feature.names
        for hf_split, ours in SPLITS.items():
            if hf_split not in ds:
                continue
            recs = [{"id": f"{lang}_{ours}_{i}", "tokens": ex["tokens"],
                     "ner_tags": decode_tags([names[t] for t in ex["ner_tags"]])}
                    for i, ex in enumerate(ds[hf_split])]
            path = os.path.join(args.out_dir, lang, f"{ours}.jsonl")
            write_jsonl(recs, path)
            print(f"{lang} {ours:<5} {len(recs):>8,} -> {path}")


if __name__ == "__main__":
    main()
