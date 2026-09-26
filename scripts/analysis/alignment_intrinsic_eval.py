#!/usr/bin/env python3
"""Intrinsic comparison of word-alignment methods on noun/entity tokens.

For a JSONL where each record holds one sentence pair and several alignment
methods (e.g. SimAlign's `itermax`, `mwmf`, `inter`, or a hybrid output), this
script reports:

  1. Noun/entity coverage per method: fraction of noun/entity words on the
     chosen side that receive at least one link.
  2. Cross-method agreement: P/R/F1 of method B's links against method A's
     links, restricted to noun/entity words (no human gold is assumed).

Noun/entity detection
  * If `--tag-key` is given, a word is an entity when its BIO tag != O.
  * Otherwise a capitalisation heuristic is applied to the ENGLISH side
    (Indic scripts have no case, so the heuristic is never applied there).
  Use --entity-side to say which side of the alignment index pairs
  (first = `--src-key`, second = `--tgt-key`) is English.

Usage:
    python scripts/analysis/alignment_intrinsic_eval.py \
        --input data/projected/mr/word2word_mr.jsonl --lang mr \
        --src-key tgt_sentence --tgt-key en_sentence --entity-side tgt \
        --methods itermax mwmf inter --output results/intrinsic/mr
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from naamsetu.languages import ALL_LANGS, lang_info  # noqa: E402

_CAP = re.compile(r"^[A-Z][a-z]+")


def load_jsonl(path, n=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


def make_indic_tok(lang):
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    code = lang_info(lang)["indic"]
    norm = IndicNormalizerFactory().get_normalizer(code)
    return lambda s: indic_tokenize.trivial_tokenize(norm.normalize(s), lang=code)


def entity_positions(words, tags=None):
    if tags:
        return {i for i, t in enumerate(tags) if t and t != "O"}
    return {i for i, w in enumerate(words) if _CAP.match(w)}


def to_pairs(raw):
    return [(int(a[0]), int(a[1])) for a in raw] if raw else []


def filter_pairs(pairs, ent_idx, side):
    k = 0 if side == "src" else 1
    return [p for p in pairs if p[k] in ent_idx]


def prf(gold, pred):
    g, p = set(gold), set(pred)
    if not g or not p:
        return 0.0, 0.0, 0.0
    tp = len(g & p)
    P, R = tp / len(p), tp / len(g)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lang", required=True, choices=ALL_LANGS)
    ap.add_argument("--src-key", default="tgt_sentence", help="sentence for the 1st index")
    ap.add_argument("--tgt-key", default="en_sentence", help="sentence for the 2nd index")
    ap.add_argument("--entity-side", choices=["src", "tgt"], default="tgt",
                    help="which side is English / carries entity tags")
    ap.add_argument("--tag-key", default=None, help="optional BIO tag list for the entity side")
    ap.add_argument("--methods", nargs="+", default=["itermax", "mwmf", "inter"])
    ap.add_argument("--max-pairs", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tok_indic = make_indic_tok(args.lang)
    tok_src = str.split if args.entity_side == "src" else tok_indic
    tok_tgt = str.split if args.entity_side == "tgt" else tok_indic

    recs = load_jsonl(args.input, args.max_pairs)
    print(f"Loaded {len(recs):,} records from {args.input}")

    cov = {m: [0, 0] for m in args.methods}           # covered, total
    cross = {(a, b): [] for a in args.methods for b in args.methods if a != b}
    for r in recs:
        s, t = r.get(args.src_key, ""), r.get(args.tgt_key, "")
        if not s or not t:
            continue
        ent_words = tok_src(s) if args.entity_side == "src" else tok_tgt(t)
        ent = entity_positions(ent_words, r.get(args.tag_key) if args.tag_key else None)
        if not ent:
            continue
        k = 0 if args.entity_side == "src" else 1
        pairs = {m: to_pairs(r.get(m)) for m in args.methods}
        for m, pr in pairs.items():
            linked = {p[k] for p in pr}
            cov[m][0] += len(ent & linked); cov[m][1] += len(ent)
        for (a, b), acc in cross.items():
            ga = filter_pairs(pairs[a], ent, args.entity_side)
            if ga:
                acc.append(prf(ga, filter_pairs(pairs[b], ent, args.entity_side)))

    print(f"\n{'Method':<15}{'Entities':>10}{'Covered':>10}{'Coverage %':>12}")
    cov_rows = []
    for m, (c, n) in cov.items():
        pct = 100 * c / n if n else 0.0
        print(f"{m:<15}{n:>10,}{c:>10,}{pct:>11.2f}%")
        cov_rows.append({"method": m, "total_entities": n, "covered": c, "coverage_pct": round(pct, 2)})

    print(f"\n{'Ref -> Pred':<25}{'P':>8}{'R':>8}{'F1':>8}{'N':>8}")
    cross_rows = []
    for (a, b), acc in cross.items():
        P, R, F = (np.mean([x[i] for x in acc]) if acc else 0.0 for i in range(3))
        print(f"{a+' -> '+b:<25}{P:>8.4f}{R:>8.4f}{F:>8.4f}{len(acc):>8,}")
        cross_rows.append({"reference": a, "pred": b, "n": len(acc),
                           "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4)})

    for name, rows in (("noun_coverage", cov_rows), ("cross_eval", cross_rows)):
        p = os.path.join(args.output, f"{name}_{args.lang}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
