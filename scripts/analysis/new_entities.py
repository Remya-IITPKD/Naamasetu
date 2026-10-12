#!/usr/bin/env python3
"""Appendix A (Table: entity strings new to Naamapadam).

An entity string is the full text of one BIO span. For each language, counts the distinct
(type, string) pairs in the silver data (data/projected/<lang>/silver_filtered_<lang>) and
how many of their strings never occur as an entity of any type in Naamapadam
train + dev + test (data/naamapadam/<lang>/, created by scripts/00_prepare_naamapadam.py).
Strings are compared exactly after NFC normalisation, so inflected or differently segmented
forms of a known name count as new.

Usage:
    python scripts/analysis/new_entities.py --out results/appendix/new_entities.csv
"""
import argparse
import csv
import gzip
import json
import os
import unicodedata

LANGS = ["ml", "ta", "te", "kn", "mr", "gu", "pa", "or", "as"]
ID2TAG = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def spans(rec):
    toks, tags = rec["tokens"], rec["ner_tags"]
    if tags and isinstance(tags[0], int):
        tags = [ID2TAG[t] for t in tags]
    out, cur = [], None
    for t, g in zip(toks, tags):
        if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur[0] != g[2:])):
            if cur:
                out.append(cur)
            cur = (g[2:], [t])
        elif g.startswith("I-"):
            cur[1].append(t)
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return [(ty, unicodedata.normalize("NFC", " ".join(w)).strip()) for ty, w in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir", default="data/projected")
    ap.add_argument("--gold-dir", default="data/naamapadam")
    ap.add_argument("--out", default="results/appendix/new_entities.csv")
    args = ap.parse_args()
    rows = []
    for L in LANGS:
        gold = set()
        for sp in ("train", "dev", "test"):
            for r in read(os.path.join(args.gold_dir, L, f"{sp}.jsonl")):
                gold.update(s for _, s in spans(r))
        silver = set()
        for r in read(os.path.join(args.silver_dir, L, f"silver_filtered_{L}.jsonl.gz")):
            silver.update(spans(r))
        new = {x for x in silver if x[1] not in gold}
        row = {"lang": L, "distinct": len(silver), "not_in_naamapadam": len(new),
               "pct_new": round(100 * len(new) / len(silver), 1)}
        for T in ("PER", "LOC", "ORG"):
            n = sum(1 for t, _ in silver if t == T)
            row[f"pct_new_{T}"] = round(100 * sum(1 for t, _ in new if t == T) / n, 1) if n else 0.0
        rows.append(row)
        print(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
