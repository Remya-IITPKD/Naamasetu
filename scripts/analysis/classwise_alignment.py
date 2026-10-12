#!/usr/bin/env python3
"""Appendix B (class-wise coverage and similarity of aligned entity words).

Coverage per class (cov_PER/LOC/ORG) and the SimAlign / awesome-align cosine per class
(cos_PER/LOC/ORG) come from results/alignment_eval/scores.csv. For the hybrid aligner the
class-wise similarity is the mean hybrid score H of its links on English entity words of
that class, read from results/alignment_eval/sample_<lang>.jsonl. The two measures differ
in scale and are not directly comparable across aligners.

Usage:
    python scripts/analysis/classwise_alignment.py --out results/appendix/classwise_alignment.csv
"""
import argparse
import csv
import json
import os

LANGS = ["ml", "mr", "kn", "gu", "pa", "ta", "te", "or", "as"]
CLS = ("PER", "LOC", "ORG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/alignment_eval")
    ap.add_argument("--out", default="results/appendix/classwise_alignment.csv")
    args = ap.parse_args()
    sc = {(r["lang"], r["method"]): r
          for r in csv.DictReader(open(os.path.join(args.eval_dir, "scores.csv"), encoding="utf-8"))}
    rows = []
    for L in LANGS:
        H = {c: [] for c in CLS}
        for line in open(os.path.join(args.eval_dir, f"sample_{L}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            for (i, _), h in zip(r["links"]["hybrid"], r["hybrid_scores"]):
                t = r["en_tags"][i]
                if t != "O":
                    H[t[2:]].append(h)
        for m in ("simalign", "awesome", "hybrid"):
            row = {"lang": L, "method": m}
            for c in CLS:
                row[f"cov_{c}"] = sc[(L, m)][f"cov_{c}"]
            for c in CLS:
                row[f"sim_{c}"] = (round(sum(H[c]) / len(H[c]), 3) if m == "hybrid"
                                   else sc[(L, m)][f"cos_{c}"])
            row["sim_measure"] = "hybrid_H" if m == "hybrid" else "cosine"
            rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
