#!/usr/bin/env python3
"""Stage 8 — Paired bootstrap significance test: augmented vs gold-only.

Reads the prediction files written by `06_evaluate.py --save-predictions`
for two models on the same test set and reports, per language, the F1 gain
(span-level by default; `--metric token` or `--metric type` for the
token-level variants, see src/naamasetu/labels.py) with a 95%
bootstrap CI and a one-sided p-value.

Usage (one --pair per language):
    python scripts/08_bootstrap_significance.py \
        --pair mr runs/eval/xlmr_gold_mr_predictions.jsonl \
                  runs/eval/xlmr_augmented_mr_predictions.jsonl \
        --pair te runs/eval/xlmr_gold_te_predictions.jsonl \
                  runs/eval/xlmr_augmented_te_predictions.jsonl \
        --out-dir results --name xlmr

Writes <out-dir>/significance_<name>.{csv,md} (`_token` / `_type` appended for the other metrics).
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.data import read_jsonl  # noqa: E402
from naamasetu.significance import paired_bootstrap  # noqa: E402


def load_predictions(path):
    rows = read_jsonl(path)
    return [r["gold"] for r in rows], [r["pred"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("LANG", "BASELINE_PRED", "SYSTEM_PRED"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--name", default="xlmr")
    ap.add_argument("--metric", choices=["span", "token", "type"], default="span")
    args = ap.parse_args()

    rows = []
    for lang, base_path, sys_path in args.pair:
        g_base, p_base = load_predictions(base_path)
        g_sys, p_sys = load_predictions(sys_path)
        if g_base != g_sys:
            raise SystemExit(f"{lang}: gold labels differ between the two files; "
                             "both must be predictions on the same test set")
        r = paired_bootstrap(g_base, p_base, p_sys, n_boot=args.n_boot, seed=args.seed,
                             metric=args.metric)
        rows.append({"lang": lang, **r})
        print(f"{lang}: gold {100 * r['f1_baseline']:.2f}  aug {100 * r['f1_system']:.2f}  "
              f"delta {100 * r['delta']:+.2f} [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}]  "
              f"p={r['p_value']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.metric == "span" else f"_{args.metric}"
    base = os.path.join(args.out_dir, f"significance_{args.name}{suffix}")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [f"{args.metric.capitalize()}-level F1 × 100; paired bootstrap, {args.n_boot} resamples; "
          "p is one-sided (H1: augmented > gold).", "",
          "| Lang | Gold F1 | Aug F1 | Δ | 95% CI | p |", "|---|---|---|---|---|---|"]
    for r in rows:
        p = "< 0.0001" if r["p_value"] == 0 else f"{r['p_value']:.4f}"
        md.append(f"| {r['lang']} | {100 * r['f1_baseline']:.2f} | {100 * r['f1_system']:.2f} | "
                  f"{100 * r['delta']:+.2f} | [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] | {p} |")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"-> {base}.csv / .md")


if __name__ == "__main__":
    main()
