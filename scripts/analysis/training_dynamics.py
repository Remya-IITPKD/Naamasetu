#!/usr/bin/env python3
"""Appendix C (training dynamics of the selected augmented runs).

Reads the per-epoch logs results/training_logs/<model>_augmented_<lang>_epoch_metrics.csv
(columns: epoch, train_loss, eval_loss, f1, ...; f1 = development F1) and reports, per run,
the epochs trained, the lowest development loss and its epoch, and the epoch kept (highest
development F1). Learning rates come from results/selected_runs.tsv.

Usage:
    python scripts/analysis/training_dynamics.py --out results/appendix/training_dynamics.csv
"""
import argparse
import csv
import os

LANGS = ["ml", "as", "mr", "or", "gu", "kn", "pa", "te", "ta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="results/training_logs")
    ap.add_argument("--selected", default="results/selected_runs.tsv")
    ap.add_argument("--out", default="results/appendix/training_dynamics.csv")
    args = ap.parse_args()
    lr = {(r["model"], r["lang"]): r["lr"]
          for r in csv.DictReader(open(args.selected, encoding="utf-8"), delimiter="\t")}
    rows = []
    for m in ("xlmr", "mbert"):
        for L in LANGS:
            log = list(csv.DictReader(open(os.path.join(
                args.log_dir, f"{m}_augmented_{L}_epoch_metrics.csv"), encoding="utf-8")))
            loss = [float(r["eval_loss"]) for r in log]
            f1 = [float(r["f1"]) for r in log]
            i, k = loss.index(min(loss)), f1.index(max(f1))
            rows.append({"model": m, "lang": L, "lr": lr[(m, L)], "epochs": len(log),
                         "min_dev_loss": round(loss[i], 3), "min_loss_epoch": int(log[i]["epoch"]),
                         "kept_epoch": int(log[k]["epoch"]), "dev_f1": round(100 * f1[k], 2)})
            print(rows[-1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
