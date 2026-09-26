#!/usr/bin/env python3
"""Stage 1 — Mine English–Indic parallel sentences from comparable Wikipedia articles.

Pipeline:
    CSV of (target title, English title) pairs
      -> fetch both Wikipedia pages -> strip non-prose markup
      -> sentence-split (indic-nlp for target, NLTK for English)
      -> multilingual MPNet sentence embeddings -> cosine similarity
      -> greedy 1-1 matching above a threshold
      -> per-page JSONL/CSV + merged JSONL/CSV

Usage:
    python scripts/01_extract_parallel.py --lang mr \
        --csv data/raw/titles/mr_en_titles_urls.csv \
        --out-dir data/parallel

Input CSV columns (auto-detected): <lang>_title|tgt_title, en_title,
optional <lang>_url|tgt_url, en_url.
Output record: {lang, tgt_title, en_title, tgt_url, en_url,
                tgt_sentence, en_sentence, cosine_similarity}
Resumable: pages whose per-page JSONL already exists are skipped.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamsetu.languages import ALL_LANGS, lang_info, script_checker  # noqa: E402

STRIP_SELECTORS = [
    "table", "infobox", "nav", "aside", "figure", "ul", "ol", ".toc",
    "blockquote", "pre", "code", ".citation", ".mw-empty-elt", ".hatnote",
    ".thumb", ".metadata", ".IPA", ".IPA-label", "span.IPA", ".mw-editsection",
    "sup", ".reference", ".noprint",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=ALL_LANGS)
    ap.add_argument("--csv", required=True, help="title/url pairs CSV")
    ap.add_argument("--out-dir", required=True, help="a <lang>/ subfolder is created")
    ap.add_argument("--model", default="paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--sim-threshold", type=float, default=0.70)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--min-sent-chars", type=int, default=10)
    ap.add_argument("--user-agent", default="naamsetu-research-crawler/0.1")
    return ap.parse_args()


def clean_text(text: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_english(text: str) -> bool:
    alpha = sum(c.isalpha() for c in text)
    ascii_alpha = sum(c.isascii() and c.isalpha() for c in text)
    return alpha > 0 and ascii_alpha / max(alpha, 1) > 0.7


def fetch_wiki_text(title, wiki_lang, timeout, ua):
    import requests
    from bs4 import BeautifulSoup
    url = f"https://{wiki_lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    r = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for sel in STRIP_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()
    paras = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    return "\n".join(p for p in paras if len(p) > 20)


def make_splitters(lang, min_chars):
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    from nltk.tokenize import sent_tokenize
    try:
        from indicnlp.tokenize.sentence_tokenize import sentence_split
    except Exception:
        sentence_split = None

    info = lang_info(lang)
    is_script = script_checker(lang)

    def split_tgt(text) -> List[str]:
        sents = []
        if sentence_split is not None:
            try:
                for para in text.split("\n"):
                    if para.strip():
                        sents.extend(sentence_split(para.strip(), info["indic"]))
            except Exception:
                sents = []
        if not sents:  # regex fallback (also covers danda punctuation)
            sents = re.split(r"(?<=[.!?।॥])\s+", text)
        return [s.strip() for s in sents if len(s.strip()) > min_chars and is_script(s)]

    def split_en(text) -> List[str]:
        return [s for s in sent_tokenize(text) if len(s) > min_chars and looks_english(s)]

    return split_tgt, split_en, sentence_split is not None


def normalise_columns(df, lang):
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    t_title = pick(f"{lang}_title", "tgt_title", "target_title", "title")
    t_url = pick(f"{lang}_url", "tgt_url", "target_url", "url")
    e_title = pick("en_title", "english_title", "eng_title")
    e_url = pick("en_url", "english_url", "eng_url")
    if t_title is None or e_title is None:
        raise ValueError(f"CSV needs target+English title columns; found {list(df.columns)}")
    ren = {t_title: "tgt_title", e_title: "en_title"}
    if t_url:
        ren[t_url] = "tgt_url"
    if e_url:
        ren[e_url] = "en_url"
    df = df.rename(columns=ren)
    for c in ("tgt_url", "en_url"):
        if c not in df.columns:
            df[c] = ""
    return df


def align_page(row, model, split_tgt, split_en, args, page_out):
    import torch
    info = lang_info(args.lang)
    slug = f"{row['tgt_title']}__{row['en_title']}".replace(" ", "_").replace("/", "_")[:180]
    out_jsonl = os.path.join(page_out, f"{slug}.jsonl")
    out_csv = os.path.join(page_out, f"{slug}.csv")
    if os.path.exists(out_jsonl):
        print(f"  skip (done): {slug}")
        return []

    tgt_sents = split_tgt(fetch_wiki_text(row["tgt_title"], info["wiki"], args.timeout, args.user_agent))
    en_sents = split_en(fetch_wiki_text(row["en_title"], "en", args.timeout, args.user_agent))
    print(f"    {args.lang} sents: {len(tgt_sents)} | EN sents: {len(en_sents)}")
    if not tgt_sents or not en_sents:
        open(out_jsonl, "w").close()
        return []

    def embed(sents):
        vecs = [model.encode(sents[i:i + args.embed_batch], convert_to_tensor=True,
                             show_progress_bar=False)
                for i in range(0, len(sents), args.embed_batch)]
        return torch.nn.functional.normalize(torch.vstack(vecs), p=2, dim=1)

    sim = embed(tgt_sents) @ embed(en_sents).T
    cands = [(float(sim[i, j]), i, j)
             for i in range(len(tgt_sents)) for j in range(len(en_sents))
             if float(sim[i, j]) >= args.sim_threshold]
    cands.sort(reverse=True, key=lambda x: x[0])

    used_t, used_e, pairs = set(), set(), []
    for score, i, j in cands:  # greedy 1-1
        if i in used_t or j in used_e:
            continue
        used_t.add(i); used_e.add(j)
        pairs.append({
            "lang": args.lang, "tgt_title": row["tgt_title"], "en_title": row["en_title"],
            "tgt_url": row.get("tgt_url", ""), "en_url": row.get("en_url", ""),
            "tgt_sentence": tgt_sents[i], "en_sentence": en_sents[j],
            "cosine_similarity": round(score, 4),
        })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if pairs:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader(); w.writerows(pairs)
    print(f"  ok {slug}: {len(pairs)} pairs")
    return pairs


def main():
    args = parse_args()
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    info = lang_info(args.lang)
    base_out = os.path.join(args.out_dir, args.lang)
    page_out = os.path.join(base_out, "pages")
    os.makedirs(page_out, exist_ok=True)
    split_tgt, split_en, indic_ok = make_splitters(args.lang, args.min_sent_chars)

    print("=" * 60)
    print(f"LANGUAGE : {args.lang} ({info['name']})")
    print(f"INPUT    : {args.csv}\nOUTPUT   : {base_out}")
    print(f"indic-nlp splitter: {'ON' if indic_ok else 'OFF (regex fallback)'}")
    print("=" * 60)

    rows = normalise_columns(pd.read_csv(args.csv), args.lang).to_dict("records")
    model = SentenceTransformer(args.model)

    merged, errors = [], []
    err_log = os.path.join(base_out, "error_log.jsonl")
    for idx, row in enumerate(rows, 1):
        print(f"\n[{idx}/{len(rows)}] {row['tgt_title']} <-> {row['en_title']}")
        try:
            merged.extend(align_page(row, model, split_tgt, split_en, args, page_out))
        except Exception as e:  # network / parse errors are logged, not fatal
            errors.append({"tgt_title": row.get("tgt_title"),
                           "en_title": row.get("en_title"), "error": str(e)})
            with open(err_log, "a", encoding="utf-8") as ef:
                ef.write(json.dumps(errors[-1], ensure_ascii=False) + "\n")

    if merged:
        mj = os.path.join(base_out, f"parallel_{args.lang}_en.jsonl")
        with open(mj, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(mj.replace(".jsonl", ".csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            w.writeheader(); w.writerows(merged)

    print("\n================ SUMMARY ================")
    print(f"Language        : {args.lang} ({info['name']})")
    print(f"Pages processed : {len(rows)}")
    print(f"Parallel pairs  : {len(merged)}")
    print(f"Errors          : {len(errors)}")


if __name__ == "__main__":
    main()
