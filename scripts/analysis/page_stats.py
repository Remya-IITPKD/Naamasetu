#!/usr/bin/env python3
"""Appendix A (Table: parallel page and sentence pairs).

For every language, counts the Wikipedia page pairs (Indic title + English title) that
yielded at least one parallel sentence in data/parallel/<lang>/, the sentence pairs, and
the minimum / maximum / average number of sentence pairs per page pair.

Usage:
    python scripts/analysis/page_stats.py --out results/appendix/page_stats.csv
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os

LANGS = ["ml", "ta", "te", "kn", "mr", "gu", "pa", "or", "as"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/parallel")
    ap.add_argument("--out", default="results/appendix/page_stats.csv")
    args = ap.parse_args()
    rows = []
    for L in LANGS:
        per_page = collections.Counter()
        for f in sorted(glob.glob(os.path.join(args.data_dir, L, f"parallel_{L}_en*.jsonl.gz"))):
            for line in gzip.open(f, "rt", encoding="utf-8"):
                r = json.loads(line)
                ek = "en_title" if "en_title" in r else "english_title"
                tk = [k for k in r if k.endswith("_title") and k != ek][0]
                per_page[(r[tk], r[ek])] += 1
        v = list(per_page.values())
        rows.append({"lang": L, "page_pairs": len(v), "sentence_pairs": sum(v),
                     "min_per_page": min(v), "max_per_page": max(v),
                     "avg_per_page": round(sum(v) / len(v), 1)})
        print(rows[-1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
