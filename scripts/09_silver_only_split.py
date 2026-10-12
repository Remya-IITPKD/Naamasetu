#!/usr/bin/env python3
"""Silver-only experiment (Appendix C): recreate the silver part of the augmented split.

Applies the same shuffle as scripts/04_merge_silver_gold.py (seed 42, 0.5% to dev) to the
silver data alone, so the silver-only models are trained on the same silver sentences as
the augmented models, without any Naamapadam training data. Writes
silver_train_<lang>.jsonl and silver_dev_<lang>.jsonl.

The models are then trained and evaluated with the standard scripts, e.g.
    python scripts/05_train_ner.py --lang mr --model-path xlm-roberta-base \
        --train data/silver_only/silver_train_mr.jsonl \
        --dev data/naamapadam/mr/dev.jsonl --test data/naamapadam/mr/test.jsonl \
        --lrs 1e-5 3e-5 5e-5 --batch-size 16 --output-dir runs/silver_only/xlmr_mr
    python scripts/06_evaluate.py --model runs/silver_only/xlmr_mr/bs16_lr<best>/final_model \
        --data data/naamapadam/mr/test.jsonl --out results/silver_only/xlmr_silveronly_mr.json

Usage:
    python scripts/09_silver_only_split.py --lang mr --out-dir data/silver_only
"""
import argparse
import gzip
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--silver-dir", default="data/projected")
    ap.add_argument("--out-dir", default="data/silver_only")
    ap.add_argument("--dev-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    L = args.lang
    random.seed(args.seed)
    silver = [json.loads(line) for line in gzip.open(
        os.path.join(args.silver_dir, L, f"silver_filtered_{L}.jsonl.gz"), "rt", encoding="utf-8")]
    random.shuffle(silver)
    n_dev = max(1, int(len(silver) * args.dev_frac))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in (("dev", silver[:n_dev]), ("train", silver[n_dev:])):
        with open(os.path.join(args.out_dir, f"silver_{name}_{L}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"tokens": r["tokens"], "ner_tags": r["ner_tags"]},
                                   ensure_ascii=False) + "\n")
        print(f"{L} {name}: {len(part):,}")


if __name__ == "__main__":
    main()
