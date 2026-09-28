#!/usr/bin/env python3
"""Stage 3b — Consistency filtering of silver data with a gold-trained model.

A model fine-tuned on gold data only tags every silver sentence. A sentence is
kept when the projected entities agree with the model's entities: span-level
F1 between the two tag sequences >= --min-agreement, and the projection has at
least one entity. Kept sentences keep their *projected* labels; the model is
used only to decide what to drop.

Usage:
    python scripts/03b_consistency_filter.py \
        --model runs/xlmr/gold/mr/seed42/bs16_lr1e-05/final_model \
        --input data/projected/mr/silver_filtered_mr.jsonl \
        --output data/projected/mr/silver_consistent_mr.jsonl

Writes the kept sentences to --output and statistics to <output>.stats.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.data import load_ner_jsonl, make_dataset, write_jsonl  # noqa: E402
from naamasetu.labels import ID2TAG, span_prf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model fine-tuned on gold data only")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path if different from --model")
    ap.add_argument("--input", required=True, help="silver NER JSONL (projected labels)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-agreement", type=float, default=0.5,
                    help="minimum sentence-level span F1 between projection and model")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        model.half()

    toks, tags = load_ner_jsonl(args.input)
    ds = make_dataset(toks, tags, tok, args.max_length)
    collate = DataCollatorForTokenClassification(tok)

    kept, scores = [], []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            batch = collate([ds[k] for k in range(i, min(i + args.batch_size, len(ds)))])
            labels = batch.pop("labels").numpy()
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            preds = np.argmax(logits.float().cpu().numpy(), axis=2)
            for j, (pr, la) in enumerate(zip(preds, labels)):
                m = la != -100
                proj = [ID2TAG[int(x)] for x in la[m]]
                pred = [ID2TAG[int(x)] for x in pr[m]]
                (_, _, f), _, _ = span_prf([proj], [pred])
                has_entity = any(t != "O" for t in proj)
                scores.append(f)
                if has_entity and f >= args.min_agreement:
                    idx = i + j
                    kept.append({"id": f"silver_{idx}", "tokens": toks[idx], "ner_tags": tags[idx]})
            if (i // args.batch_size) % 200 == 0:
                print(f"  {i + len(preds):>8,}/{len(ds):,}  kept {len(kept):,}", flush=True)

    write_jsonl(kept, args.output)
    stats = {"input": len(toks), "kept": len(kept), "kept_fraction": len(kept) / max(len(toks), 1),
             "min_agreement": args.min_agreement, "mean_agreement": float(np.mean(scores)) if scores else 0.0}
    json.dump(stats, open(args.output + ".stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
