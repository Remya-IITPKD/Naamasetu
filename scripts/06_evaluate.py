#!/usr/bin/env python3
"""Stage 6 — Evaluate a saved model on any NER JSONL file (inference only).

Reports three metrics, strictest first:
  span_*   span-level (CoNLL exact match) P/R/F1, overall and per type
  token_*  token-level micro P/R/F1 over the non-O BIO tags
  type_*   token-level micro P/R/F1 after merging B-/I- into PER/LOC/ORG,
           overall and per type (the "entity_micro_F1" of the training logs)
No model selection happens here; select checkpoints on dev during training.

Usage:
    python scripts/06_evaluate.py \
        --model runs/xlmr/augmented/mr/seed42/bs16_lr3e-05/final_model \
        --data  data/augmented/mr/test_mr.jsonl \
        --out   runs/xlmr/augmented/mr/seed42/eval_test.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.data import load_ner_jsonl, make_dataset  # noqa: E402
from naamasetu.labels import ENTITY_TYPES, ID2TAG, span_prf, token_prf, type_token_prf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="load the tokenizer from here instead of --model (e.g. the base "
                         "model, if the saved tokenizer.json needs a newer `tokenizers`)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="JSON file for metrics")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    toks, tags = load_ner_jsonl(args.data)
    ds = make_dataset(toks, tags, tok, args.max_length)
    collate = DataCollatorForTokenClassification(tok)

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            batch = collate([ds[k] for k in range(i, min(i + args.batch_size, len(ds)))])
            labels = batch.pop("labels").numpy()
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            preds = np.argmax(logits.cpu().numpy(), axis=2)
            for pr, la in zip(preds, labels):
                m = la != -100
                y_true.append([ID2TAG[int(x)] for x in la[m]])
                y_pred.append([ID2TAG[int(x)] for x in pr[m]])

    (p, r, f), per_type, report = span_prf(y_true, y_pred)
    tp_, tr_, tf_ = token_prf(y_true, y_pred)
    (yp, yr, yf), per_type_tok = type_token_prf(y_true, y_pred)
    out = {"model": args.model, "data": args.data, "n_sentences": len(y_true),
           "span_p": p, "span_r": r, "span_f1": f,
           "token_p": tp_, "token_r": tr_, "token_f1": tf_,
           "type_p": yp, "type_r": yr, "type_f1": yf,
           **{f"type_{e}_{k}": per_type_tok[e][i] for e in ENTITY_TYPES
              for i, k in enumerate(["p", "r", "f1", "support"])},
           **{f"{e}_{k}": per_type[e][i] for e in ENTITY_TYPES
              for i, k in enumerate(["p", "r", "f1", "support"])}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    if args.save_predictions:
        with open(args.out.replace(".json", "_predictions.jsonl"), "w", encoding="utf-8") as fh:
            for t, g, pr in zip(toks, y_true, y_pred):
                fh.write(json.dumps({"tokens": t[:len(g)], "gold": g, "pred": pr},
                                    ensure_ascii=False) + "\n")
    print(report)
    print(f"\nspan F1 = {f:.4f}  token F1 = {tf_:.4f}  type F1 = {yf:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
