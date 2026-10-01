#!/usr/bin/env python3
"""Alignment quality of SimAlign, awesome-align and the Naamasetu hybrid aligner on the
samples written by build_alignment_sample.py.

Per language and aligner:
  src_cov      fraction of English words with >= 1 link
  tgt_cov      fraction of target words with >= 1 link
  ent_cov      fraction of English entity words (spaCy PER/LOC/ORG) with >= 1 link
  ent_cos      mean cosine similarity of the links whose English word is an entity word,
               computed for every aligner with the SAME encoder (word vector = mean of the
               sub-word states of one hidden layer, as in 02_project_labels.py); only with
               --model
  span_any     fraction of English entity spans (BIO) with >= 1 linked word
  span_all     fraction of English entity spans whose words are all linked
  span_cos     mean, over spans with >= 1 link, of the cosine between the mean vector of the
               span's English words and the mean vector of the target words linked to them
               (same encoder; only with --model)
  cov_<T>, cos_<T>   entity-word coverage and entity-word cosine per type T in PER/LOC/ORG
  hybrid_H     (hybrid only) mean of the hybrid score H stored in the file on those links;
               H is not a cosine (it mixes Sem/Pho/Roman, adds the entity bonus and is
               normalised per sentence), so it is reported separately and not compared.

Usage:
    python scripts/analysis/alignment_eval_scores.py --sample-dir results/alignment_eval \
        [--model bert-base-multilingual-cased --layer 8] --out results/alignment_eval/scores.csv
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

METHODS = ("simalign", "awesome", "hybrid")
TYPES = ("PER", "LOC", "ORG")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Embedder:
    def __init__(self, model, layer, offline):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.layer = torch, layer
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=offline)
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True,
                                               local_files_only=offline).to(self.dev).eval()

    def __call__(self, words):
        with self.torch.no_grad():
            inp = self.tok(words, is_split_into_words=True, return_tensors="pt",
                           truncation=True).to(self.dev)
            hs = self.model(**inp).hidden_states[self.layer][0]
        wids = inp.word_ids(0)
        out = np.full((len(words), hs.shape[-1]), np.nan, dtype=np.float32)
        for w in set(x for x in wids if x is not None):
            out[w] = hs[[k for k, x in enumerate(wids) if x == w]].mean(0).cpu().numpy()
        return out


def spans(tags):
    """[(start, end_exclusive)] of BIO entity spans."""
    out, start = [], None
    for k, t in enumerate(tags + ["O"]):
        if start is not None and not t.startswith("I-"):
            out.append((start, k)); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            start = k
    return out


def cos(a, b):
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default="results/alignment_eval")
    ap.add_argument("--model", default=None, help="encoder for ent_cos (omit to skip)")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="results/alignment_eval/scores.csv")
    args = ap.parse_args()

    emb = Embedder(args.model, args.layer, args.offline) if args.model else None
    rows = []
    for path in sorted(glob.glob(os.path.join(args.sample_dir, "sample_*.jsonl"))):
        recs = load(path)
        lang = recs[0]["lang"]
        acc = {m: dict(src=0, tgt=0, ent=0, cos=[], H=[], s_any=0, s_all=0, s_cos=[],
                       **{f"c_{T}": 0 for T in TYPES}, **{f"k_{T}": [] for T in TYPES}) for m in METHODS}
        n_src = n_tgt = n_ent = n_span = 0
        n_type = {T: 0 for T in TYPES}
        for r in recs:
            ew, tw, tags = r["en_words"], r["tgt_words"], r["en_tags"]
            ent = {i for i, t in enumerate(tags) if t != "O"}
            sp = spans(tags)
            etype = {i: t[2:] for i, t in enumerate(tags) if t != "O"}
            for T in etype.values():
                n_type[T] += 1
            n_src += len(ew); n_tgt += len(tw); n_ent += len(ent); n_span += len(sp)
            U = emb(ew) if emb else None
            V = emb(tw) if emb else None
            for m in METHODS:
                links = r["links"][m]
                si, tj = {i for i, _ in links}, {j for _, j in links}
                acc[m]["src"] += len(si); acc[m]["tgt"] += len(tj); acc[m]["ent"] += len(si & ent)
                for i in si & ent:
                    acc[m][f"c_{etype[i]}"] += 1
                if emb:
                    for i, j in links:
                        if i in ent:
                            c = cos(U[i], V[j])
                            if c is not None:
                                acc[m][f"k_{etype[i]}"].append(c)
                for a, b in sp:
                    words = set(range(a, b))
                    linked = words & si
                    acc[m]["s_any"] += bool(linked); acc[m]["s_all"] += linked == words
                    tgt_j = sorted({j for i, j in links if i in words})
                    if emb and tgt_j:
                        e = np.nanmean(U[a:b], axis=0); t = np.nanmean(V[tgt_j], axis=0)
                        c = cos(e, t)
                        if c is not None:
                            acc[m]["s_cos"].append(c)
                for k, (i, j) in enumerate(links):
                    if i in ent:
                        if emb:
                            c = cos(U[i], V[j])
                            if c is not None:
                                acc[m]["cos"].append(c)
                        if m == "hybrid":
                            acc[m]["H"].append(r["hybrid_scores"][k])
        for m in METHODS:
            a = acc[m]
            rows.append({"lang": lang, "method": m, "pairs": len(recs),
                         "src_cov": round(a["src"] / n_src, 3), "tgt_cov": round(a["tgt"] / n_tgt, 3),
                         "ent_cov": round(a["ent"] / n_ent, 3) if n_ent else "",
                         "ent_links": len(a["cos"]) if emb else "",
                         "ent_cos": round(float(np.mean(a["cos"])), 3) if a["cos"] else "",
                         "span_any": round(a["s_any"] / n_span, 3) if n_span else "",
                         "span_all": round(a["s_all"] / n_span, 3) if n_span else "",
                         "span_cos": round(float(np.mean(a["s_cos"])), 3) if a["s_cos"] else "",
                         **{f"cov_{T}": round(a[f"c_{T}"] / n_type[T], 3) if n_type[T] else "" for T in TYPES},
                         **{f"cos_{T}": round(float(np.mean(a[f"k_{T}"])), 3) if a[f"k_{T}"] else "" for T in TYPES},
                         **{f"n_{T}": n_type[T] for T in TYPES},
                         "hybrid_H": round(float(np.mean(a["H"])), 3) if a["H"] else ""})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    print("written", args.out)


if __name__ == "__main__":
    main()
