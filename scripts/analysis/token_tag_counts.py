#!/usr/bin/env python3
"""How is a word tagged in an NER corpus?

Counts the BIO tags of every token that starts with PREFIX (so inflected forms
are included), broken down by token form, and prints the first matching
sentences with their tags.

Usage:
    python scripts/analysis/token_tag_counts.py data/naamapadam/ml/train.jsonl വിഷു
    python scripts/analysis/token_tag_counts.py data/projected/ml/silver_filtered_ml.jsonl.gz വിഷു --show 3
"""

import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from naamasetu.labels import decode_tags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="NER JSONL (tokens + ner_tags), plain or .gz")
    ap.add_argument("prefix", help="count tokens starting with this string")
    ap.add_argument("--show", type=int, default=5, help="matching sentences to print")
    args = ap.parse_args()

    opener = gzip.open if args.path.endswith(".gz") else open
    total = collections.Counter()
    by_form = collections.defaultdict(collections.Counter)
    shown = 0
    with opener(args.path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if args.prefix not in line:
                continue
            r = json.loads(line)
            toks, tags = r["tokens"], decode_tags(r["ner_tags"])
            hit = False
            for t, y in zip(toks, tags):
                if t.startswith(args.prefix):
                    total[y] += 1
                    by_form[t][y] += 1
                    hit = True
            if hit and shown < args.show:
                shown += 1
                print(f"line {lineno}:")
                print("  Tokens :", toks)
                print("  Tags   :", tags)
                for t, y in zip(toks, tags):
                    if y != "O" or t.startswith(args.prefix):
                        print(f"  {'ENTITY' if y != 'O' else 'other '} : {t} -> {y}")

    n = sum(total.values())
    print(f"\n{n} tokens starting with {args.prefix!r} in {args.path}")
    for y, c in total.most_common():
        print(f"  {y:<6} {c:>6}  ({100 * c / n:.1f}%)")
    print("\nby token form:")
    for t, c in sorted(by_form.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {t}  " + ", ".join(f"{y} {k}" for y, k in c.most_common()))


if __name__ == "__main__":
    main()
