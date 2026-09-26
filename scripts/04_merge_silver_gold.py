#!/usr/bin/env python3
"""Stage 4 — Merge filtered silver data into the Naamapadam gold splits.

* Silver is shuffled and split (1 - dev_frac) -> train, dev_frac -> dev.
* The gold TEST split is copied through untouched.
* Sentence / entity / tag statistics are printed and saved.

Usage:
    python scripts/04_merge_silver_gold.py --lang mr \
        --silver data/projected/mr/silver_filtered_mr.jsonl \
        --train  data/naamapadam/mr/train.jsonl \
        --dev    data/naamapadam/mr/dev.jsonl \
        --test   data/naamapadam/mr/test.jsonl \
        --out-dir data/augmented/mr

Outputs: train_aug_<lang>.jsonl, dev_aug_<lang>.jsonl, test_<lang>.jsonl,
         stats_<lang>.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.data import read_jsonl, write_jsonl  # noqa: E402
from naamasetu.labels import decode_tags  # noqa: E402


def stats(records):
    ent, tags = Counter(), Counter()
    for r in records:
        t = decode_tags(r.get("ner_tags", []))
        tags.update(t)
        ent.update(x[2:] for x in t if x.startswith("B-"))
    return {"sentences": len(records), "entities": dict(ent), "tags": dict(tags)}


def show(label, s):
    print(f"\n  [{label}]\n    Sentences : {s['sentences']:,}")
    for e in ("PER", "LOC", "ORG", "MISC"):
        if s["entities"].get(e):
            print(f"    {e:<6}    : {s['entities'][e]:,}")
    print("    Tag breakdown:")
    for t in sorted(s["tags"]):
        print(f"      {t:<12} {s['tags'][t]:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-frac", type=float, default=0.005,
                    help="fraction of silver added to dev (default 0.5%%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    silver = read_jsonl(args.silver)
    g_train, g_dev, g_test = read_jsonl(args.train), read_jsonl(args.dev), read_jsonl(args.test)
    print(f"Silver {len(silver):,} | gold train {len(g_train):,} "
          f"dev {len(g_dev):,} test {len(g_test):,}")

    random.shuffle(silver)
    n_dev = max(1, int(len(silver) * args.dev_frac))
    s_dev, s_train = silver[:n_dev], silver[n_dev:]

    m_train, m_dev, m_test = g_train + s_train, g_dev + s_dev, g_test
    random.shuffle(m_train)
    random.shuffle(m_dev)

    L = args.lang
    write_jsonl(m_train, os.path.join(args.out_dir, f"train_aug_{L}.jsonl"))
    write_jsonl(m_dev, os.path.join(args.out_dir, f"dev_aug_{L}.jsonl"))
    write_jsonl(m_test, os.path.join(args.out_dir, f"test_{L}.jsonl"))

    report = {
        "lang": L, "seed": args.seed, "dev_frac": args.dev_frac,
        "counts": {"silver_total": len(silver), "silver_train": len(s_train),
                   "silver_dev": len(s_dev), "gold_train": len(g_train),
                   "gold_dev": len(g_dev), "gold_test": len(g_test)},
        "train": stats(m_train), "dev": stats(m_dev), "test": stats(m_test),
    }
    report["total"] = stats(m_train + m_dev + m_test)
    for k in ("train", "dev", "test", "total"):
        show(k.upper(), report[k])
    with open(os.path.join(args.out_dir, f"stats_{L}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
