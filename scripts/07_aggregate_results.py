#!/usr/bin/env python3
"""Stage 7 — Aggregate results.json files across seeds into paper tables.

Expects the layout produced by slurm/train.slurm:
    runs/<model>/<setting>/<lang>/seed<S>/results.json
where <setting> is e.g. `gold` (baseline) or `augmented` (gold + silver).

Writes mean ± std (over seeds; just the score when there is one seed) of test span F1 per model/setting/language,
plus the per-language delta (augmented - gold), as CSV, Markdown and LaTeX.

Usage:
    python scripts/07_aggregate_results.py --runs-dir runs --out-dir results
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

LANG_ORDER = ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--metric", default="test_span_f1")
    args = ap.parse_args()

    scores = defaultdict(list)  # (model, setting, lang) -> [f1 per seed]
    for path in glob.glob(os.path.join(args.runs_dir, "*", "*", "*", "seed*", "results.json")):
        model, setting, lang = path.split(os.sep)[-5:-2]
        scores[(model, setting, lang)].append(json.load(open(path))[args.metric] * 100)

    if not scores:
        raise SystemExit(f"No results.json found under {args.runs_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    models = sorted({k[0] for k in scores})
    found = {k[1] for k in scores}
    settings = [x for x in ("gold", "augmented") if x in found] + sorted(found - {"gold", "augmented"})
    langs = [l for l in LANG_ORDER if any(k[2] == l for k in scores)]

    rows = []
    for m in models:
        for s in settings:
            for l in langs:
                v = scores.get((m, s, l))
                if v:
                    rows.append({"model": m, "setting": s, "lang": l, "n_seeds": len(v),
                                 "mean": mean(v), "std": pstdev(v) if len(v) > 1 else 0.0})

    with open(os.path.join(args.out_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("model,setting,lang,n_seeds,mean,std\n")
        for r in rows:
            f.write(f"{r['model']},{r['setting']},{r['lang']},{r['n_seeds']},{r['mean']:.2f},{r['std']:.2f}\n")

    idx = {(r["model"], r["setting"], r["lang"]): r for r in rows}
    cell = lambda r: ("–" if not r else f"{r['mean']:.2f}" if r["n_seeds"] == 1
                      else f"{r['mean']:.2f} ± {r['std']:.2f}")
    md = ["| Model | Setting | " + " | ".join(langs) + " | Avg |",
          "|---|---|" + "---|" * (len(langs) + 1)]
    tex = [r"\begin{tabular}{ll" + "c" * (len(langs) + 1) + "}", r"\toprule",
           "Model & Setting & " + " & ".join(langs) + r" & Avg \\", r"\midrule"]
    for m in models:
        for s in settings:
            rs = [idx.get((m, s, l)) for l in langs]
            if not any(rs):
                continue
            avg = mean(r["mean"] for r in rs if r)
            md.append(f"| {m} | {s} | " + " | ".join(cell(r) for r in rs) + f" | {avg:.2f} |")
            tex.append(f"{m} & {s} & " + " & ".join(
                ("--" if not r else f"{r['mean']:.1f}" if r["n_seeds"] == 1
                 else f"{r['mean']:.1f}$_{{\\pm{r['std']:.1f}}}$") for r in rs)
                + f" & {avg:.1f} \\\\")
        if "gold" in settings and "augmented" in settings:
            d = [(idx[(m, "augmented", l)]["mean"] - idx[(m, "gold", l)]["mean"])
                 if (m, "augmented", l) in idx and (m, "gold", l) in idx else None for l in langs]
            if any(x is not None for x in d):
                md.append(f"| {m} | Δ | " + " | ".join(f"{x:+.2f}" if x is not None else "–" for x in d) + " | |")
    tex += [r"\bottomrule", r"\end{tabular}"]

    open(os.path.join(args.out_dir, "main_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(args.out_dir, "main_table.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
