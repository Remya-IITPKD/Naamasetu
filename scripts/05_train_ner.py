#!/usr/bin/env python3
"""Stage 5 — Fine-tune mBERT or XLM-R for NER on gold or gold+silver data.

* One script for both encoders (select with --model-path).
* Relative-loss early stopping on dev loss (see RelativeLossEarlyStopping).
* Optional LR sweep; the best LR is selected on DEV F1 only.
* The chosen model is evaluated once on TEST (span-level, CoNLL-style).

Usage (paper setting, one seed):
    python scripts/05_train_ner.py --lang mr \
        --model-path xlm-roberta-base \
        --train data/augmented/mr/train_aug_mr.jsonl \
        --dev   data/augmented/mr/dev_aug_mr.jsonl \
        --test  data/augmented/mr/test_mr.jsonl \
        --lrs 3e-5 --seed 42 \
        --output-dir runs/xlmr/augmented/mr/seed42

Outputs in --output-dir:
    bs<B>_lr<LR>/final_model/      best checkpoint (by dev F1) per LR
    bs<B>_lr<LR>/epoch_metrics.csv train/dev loss and dev P/R/F1/acc per epoch
    all_lrs_summary.csv            dev F1 / loss / best epoch per LR
    training_curves.json           all curves (for plotting)
    results.json                   machine-readable final numbers (dev + test)
    test_report.txt                span-level and token-level test reports
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamsetu.data import load_ner_jsonl, make_dataset  # noqa: E402
from naamsetu.labels import ENTITY_TYPES, ID2TAG, TAG2ID, TAG_LIST, span_prf  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--model-path", required=True,
                    help="bert-base-multilingual-cased | xlm-roberta-base | local path")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-5],
                    help="paper sweep: 1e-5 1e-6 3e-5 3e-6 5e-5 5e-6 (3e-5 selected)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--rel-tol", type=float, default=0.005,
                    help="min relative dev-loss improvement to reset patience (0.5%%)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def make_early_stopping(TrainerCallback):
    class RelativeLossEarlyStopping(TrainerCallback):
        """Stop when dev loss fails to improve by >= rel_tol (relative) for
        `patience` consecutive evaluations. More robust than absolute-delta
        patience when loss scales differ across languages / data sizes."""

        def __init__(self, rel_tol=0.005, patience=3):
            self.rel_tol, self.patience = rel_tol, patience
            self.reset()

        def reset(self):
            self._best, self._bad = float("inf"), 0

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics is None:
                return
            cur = metrics.get("eval_loss", float("inf"))
            if self._best == float("inf"):
                self._best = cur
                return
            rel = (self._best - cur) / (self._best + 1e-12)
            if rel >= self.rel_tol:
                self._best, self._bad = cur, 0
                print(f"  [ES] epoch {round(state.epoch)} loss={cur:.5f} rel_imp={rel*100:.3f}% ok")
            else:
                self._bad += 1
                print(f"  [ES] epoch {round(state.epoch)} loss={cur:.5f} rel_imp={rel*100:.3f}% "
                      f"bad={self._bad}/{self.patience}")
                if self._bad >= self.patience:
                    print("  [ES] Stopping.")
                    control.should_training_stop = True

    return RelativeLossEarlyStopping


def token_metrics(p):
    """Token-level micro P/R/F1 over entity tags (O excluded) + accuracy.
    Used for checkpoint selection during training."""
    from sklearn.metrics import precision_recall_fscore_support
    preds = np.argmax(p.predictions, axis=2)
    yp, yt = [], []
    for pr, la in zip(preds, p.label_ids):
        for pi, li in zip(pr, la):
            if li != -100:
                yp.append(pi); yt.append(li)
    ent = [i for i, t in enumerate(TAG_LIST) if t != "O"]
    P, R, F, _ = precision_recall_fscore_support(yt, yp, average="micro", labels=ent, zero_division=0)
    return {"precision": P, "recall": R, "f1": F, "accuracy": float((np.array(yt) == np.array(yp)).mean())}


def training_args(TrainingArguments, **kw):
    """Handle the evaluation_strategy -> eval_strategy rename across transformers versions."""
    try:
        return TrainingArguments(eval_strategy="epoch", **kw)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kw)


def make_trainer(Trainer, tok, **kw):
    try:
        return Trainer(processing_class=tok, **kw)
    except TypeError:
        return Trainer(tokenizer=tok, **kw)


def decode_predictions(pred):
    preds = np.argmax(pred.predictions, axis=2)
    y_true, y_pred = [], []
    for pr, la in zip(preds, pred.label_ids):
        t, p = [], []
        for pi, li in zip(pr, la):
            if li != -100:
                t.append(ID2TAG[int(li)]); p.append(ID2TAG[int(pi)])
        y_true.append(t); y_pred.append(p)
    return y_true, y_pred


def main():
    args = parse_args()
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print(f"Language : {args.lang}\nModel    : {args.model_path}\nSeed     : {args.seed}")
    print(f"Output   : {args.output_dir}")
    print(f"Early stop: rel_tol={args.rel_tol*100:.2f}% patience={args.patience}")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading data ...")
    tr, dv, te = (load_ner_jsonl(p) for p in (args.train, args.dev, args.test))
    train_ds, dev_ds, test_ds = (make_dataset(*d, tok, args.max_length) for d in (tr, dv, te))
    collator = DataCollatorForTokenClassification(tok)
    es = make_early_stopping(TrainerCallback)(args.rel_tol, args.patience)

    summary, curves = [], {}
    for lr in args.lrs:
        run = os.path.join(args.output_dir, f"bs{args.batch_size}_lr{lr:g}")
        os.makedirs(run, exist_ok=True)
        res_path = os.path.join(run, "combo_result.json")
        if os.path.exists(res_path) and os.path.isdir(os.path.join(run, "final_model")):
            rec = json.load(open(res_path)); summary.append(rec)
            ep_csv = os.path.join(run, "epoch_metrics.csv")
            if os.path.exists(ep_csv):
                curves[f"lr{lr:g}"] = list(csv.DictReader(open(ep_csv)))
            print(f"SKIP lr={lr:g} (done) dev_f1={rec['dev_f1']:.4f}")
            continue

        print(f"\n{'='*55}\nLR = {lr:g}\n{'='*55}")
        set_seed(args.seed); es.reset()
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, num_labels=len(TAG_LIST), id2label=ID2TAG,
            label2id=TAG2ID, ignore_mismatched_sizes=True)
        targs = training_args(
            TrainingArguments,
            output_dir=os.path.join(run, "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=32,
            learning_rate=lr,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_strategy="epoch",
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=2,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
        )
        trainer = make_trainer(Trainer, tok, model=model, args=targs,
                               train_dataset=train_ds, eval_dataset=dev_ds,
                               data_collator=collator, compute_metrics=token_metrics,
                               callbacks=[es])
        trainer.train()

        rows = {}
        for e in trainer.state.log_history:
            if e.get("epoch") is None:
                continue
            r = rows.setdefault(round(e["epoch"]), {"epoch": round(e["epoch"])})
            if "loss" in e:
                r["train_loss"] = round(e["loss"], 6)
            if "eval_loss" in e:
                r.update(eval_loss=round(e["eval_loss"], 6),
                         f1=round(e.get("eval_f1", 0), 6),
                         precision=round(e.get("eval_precision", 0), 6),
                         recall=round(e.get("eval_recall", 0), 6),
                         accuracy=round(e.get("eval_accuracy", 0), 6))
        fields = ["epoch", "train_loss", "eval_loss", "f1", "precision", "recall", "accuracy"]
        with open(os.path.join(run, "epoch_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for ep in sorted(rows):
                w.writerow(rows[ep])
        curves[f"lr{lr:g}"] = [rows[ep] for ep in sorted(rows)]

        dev_m = trainer.evaluate(dev_ds)
        ev = [(r["eval_loss"], r["epoch"]) for r in rows.values() if "eval_loss" in r]
        best_ep = min(ev)[1] if ev else None
        trainer.save_model(os.path.join(run, "final_model"))
        tok.save_pretrained(os.path.join(run, "final_model"))

        rec = {"lang": args.lang, "lr": lr, "dev_f1": dev_m["eval_f1"],
               "dev_loss": dev_m["eval_loss"], "best_epoch": best_ep, "run": run}
        json.dump(rec, open(res_path, "w"), indent=2)
        summary.append(rec)
        print(f"lr={lr:g} dev_f1={rec['dev_f1']:.4f} dev_loss={rec['dev_loss']:.4f} best_epoch={best_ep}")
        del trainer, model
        torch.cuda.empty_cache()

    json.dump({"lang": args.lang, "model": args.model_path, "lrs": args.lrs,
               "batch_size": args.batch_size, "seed": args.seed, "curves": curves},
              open(os.path.join(args.output_dir, "training_curves.json"), "w"), indent=2)
    with open(os.path.join(args.output_dir, "all_lrs_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lr", "dev_f1", "dev_loss", "best_epoch"]); w.writeheader()
        for r in summary:
            w.writerow({k: r[k] for k in ["lr", "dev_f1", "dev_loss", "best_epoch"]})

    # ---- model selection on DEV only; single evaluation on TEST ----
    best = max(summary, key=lambda r: r["dev_f1"])
    print(f"\nBest LR (by dev F1) = {best['lr']:g}  dev_f1 = {best['dev_f1']:.4f}")
    model = AutoModelForTokenClassification.from_pretrained(os.path.join(best["run"], "final_model"))
    ev_tr = make_trainer(Trainer, tok, model=model,
                         args=TrainingArguments(output_dir=os.path.join(args.output_dir, "_tmp"),
                                                per_device_eval_batch_size=64, report_to="none"),
                         data_collator=collator, compute_metrics=token_metrics)
    pred = ev_tr.predict(test_ds)
    y_true, y_pred = decode_predictions(pred)
    (sp, sr, sf), per_type, span_report = span_prf(y_true, y_pred)
    tok_m = {k.replace("test_", ""): v for k, v in pred.metrics.items()
             if k.split("_")[-1] in ("precision", "recall", "f1", "accuracy")}

    results = {
        "lang": args.lang, "model": args.model_path, "seed": args.seed,
        "best_lr": best["lr"], "best_epoch": best["best_epoch"], "dev_token_f1": best["dev_f1"],
        "test_span_p": sp, "test_span_r": sr, "test_span_f1": sf,
        **{f"test_span_{e}_f1": per_type[e][2] for e in ENTITY_TYPES},
        **{f"test_token_{k}": v for k, v in tok_m.items()},
    }
    json.dump(results, open(os.path.join(args.output_dir, "results.json"), "w"), indent=2)
    with open(os.path.join(args.output_dir, "test_report.txt"), "w") as f:
        f.write(f"lang={args.lang} model={args.model_path} seed={args.seed} "
                f"lr={best['lr']:g} best_epoch={best['best_epoch']}\n\n")
        f.write("=== Span-level (exact match, CoNLL) ===\n" + span_report + "\n\n")
        f.write("=== Token-level entity micro (O excluded) ===\n"
                + json.dumps(tok_m, indent=2) + "\n")

    print(f"\n{'='*55}\nFINAL — {args.lang}\n{'='*55}")
    print(f"Dev token F1   : {best['dev_f1']:.4f}")
    print(f"Test span F1   : {sf:.4f}  (P={sp:.4f} R={sr:.4f})")
    print(span_report)


if __name__ == "__main__":
    main()
