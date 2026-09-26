#!/usr/bin/env python3
"""Alignment quality statistics for SimAlign / awesome-align / hybrid.

Per method, over a corpus:
    sents            number of sentence pairs
    pairs            total aligned (src, tgt) links
    avg_similarity   mean alignment score over links
    source_coverage  fraction of SOURCE words with >= 1 link
    target_coverage  fraction of TARGET words with >= 1 link

Accepted JSONL shapes (auto-detected):
  (A) index form (written by 02_project_labels.py):
      {"en_sentence": ..., "tgt_sentence": ...,
       "alignments": [{"src_idx": i, "tgt_idx": j, "score": s}, ...]}
      -> word counts come from re-tokenising the sentences (needs --lang).
  (B) pair form (SimAlign / awesome-align dumps):
      {"source": [...], "target": [...], "alignments": [[i, j], ...]}
      or "aligns": "i-j i-j ..."   (score defaults to 1.0)

Usage:
    python scripts/analysis/alignment_coverage_stats.py --lang mr \
        simalign=align_simalign_mr.jsonl awesome=align_awesome_mr.jsonl \
        hybrid=data/projected/mr/alignment_results_mr.jsonl --csv results/align_stats_mr.csv
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from naamsetu.languages import ALL_LANGS, lang_info  # noqa: E402


def make_tokenizers(lang):
    tok_en = str.split
    if not lang:
        return tok_en, str.split
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
    except Exception:
        return tok_en, str.split
    code = lang_info(lang)["indic"]
    try:
        norm = IndicNormalizerFactory().get_normalizer(code)
    except Exception:
        norm = None
    return tok_en, lambda s: indic_tokenize.trivial_tokenize(norm.normalize(s) if norm else s, lang=code)


def parse_alignments(rec):
    al = rec.get("alignments", rec.get("aligns", rec.get("alignment", [])))
    out = []
    if isinstance(al, str):
        for t in al.split():
            if "-" in t:
                a, b = t.split("-")[:2]
                try:
                    out.append((int(a), int(b), 1.0))
                except ValueError:
                    pass
        return out
    for a in al or []:
        if isinstance(a, dict):
            i = a.get("src_idx", a.get("i", a.get("source")))
            j = a.get("tgt_idx", a.get("j", a.get("target")))
            if i is not None and j is not None:
                out.append((int(i), int(j), float(a.get("score", a.get("sim", 1.0)))))
        elif isinstance(a, (list, tuple)) and len(a) >= 2:
            out.append((int(a[0]), int(a[1]), float(a[2]) if len(a) >= 3 else 1.0))
    return out


def n_words(rec, tok_en, tok_tgt):
    src = rec.get("source") or rec.get("src_tokens") or rec.get("en_tokens")
    tgt = rec.get("target") or rec.get("tgt_tokens")
    if isinstance(src, list) and isinstance(tgt, list):
        return len(src), len(tgt)
    en = rec.get("en_sentence", rec.get("source_sentence", ""))
    tg = rec.get("tgt_sentence", rec.get("target_sentence", ""))
    return len(tok_en(en)), len(tok_tgt(tg))


def score_file(path, tok_en, tok_tgt):
    st = dict(sents=0, pairs=0, score_sum=0.0, src_al=0, tgt_al=0, src_total=0, tgt_total=0)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st["sents"] += 1
            ns, nt = n_words(rec, tok_en, tok_tgt)
            st["src_total"] += ns; st["tgt_total"] += nt
            si, tj = set(), set()
            for i, j, s in parse_alignments(rec):
                st["pairs"] += 1; st["score_sum"] += s; si.add(i); tj.add(j)
            st["src_al"] += sum(1 for i in si if ns == 0 or i < ns)
            st["tgt_al"] += sum(1 for j in tj if nt == 0 or j < nt)
    return {
        "sents": st["sents"], "pairs": st["pairs"],
        "avg_similarity": st["score_sum"] / st["pairs"] if st["pairs"] else 0.0,
        "source_coverage": st["src_al"] / st["src_total"] if st["src_total"] else 0.0,
        "target_coverage": st["tgt_al"] / st["tgt_total"] if st["tgt_total"] else 0.0,
        "src_total": st["src_total"], "tgt_total": st["tgt_total"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=ALL_LANGS, default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("methods", nargs="+", help="name=path entries")
    args = ap.parse_args()

    tok_en, tok_tgt = make_tokenizers(args.lang)
    rows = []
    for spec in args.methods:
        name, path = spec.split("=", 1) if "=" in spec else \
            (os.path.splitext(os.path.basename(spec))[0], spec)
        if not os.path.exists(path):
            print(f"!! missing file for '{name}': {path}", file=sys.stderr); continue
        rows.append({"method": name, **score_file(path, tok_en, tok_tgt)})
    if not rows:
        sys.exit("No valid method files.")

    hdr = ["method", "sents", "pairs", "avg_similarity", "source_coverage", "target_coverage"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    w = {h: max(len(h), *(len(fmt(r[h])) for r in rows)) for h in hdr}
    print(f"\nLanguage: {args.lang or '(token-list inputs)'}")
    print("  ".join(h.ljust(w[h]) for h in hdr)); print("-" * (sum(w.values()) + 2 * len(hdr)))
    for r in rows:
        print("  ".join(fmt(r[h]).ljust(w[h]) for h in hdr))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=hdr + ["src_total", "tgt_total"]); cw.writeheader()
            cw.writerows(rows)
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
