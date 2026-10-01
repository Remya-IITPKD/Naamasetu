#!/usr/bin/env python3
"""Draw a fixed random sample of sentence pairs aligned by all three aligners
(SimAlign, awesome-align, Naamasetu hybrid) and put their links on a common footing.

For every language the script
  1. reads the released files in data/parallel/<lang>/ and data/projected/<lang>/
     (gzipped, possibly split into .partN files);
  2. keeps the pairs present in all three alignment files and samples N of them
     with a fixed seed;
  3. splits both sentences into whitespace words and maps every link of every
     aligner onto these words:
       * SimAlign / awesome-align store links as word strings; each is mapped to the
         first not-yet-linked whitespace word with that string;
       * the released hybrid files store word indices with the two fields swapped:
         `src_idx` indexes the Indic-NLP tokens of the (normalised) target sentence and
         `tgt_idx` indexes en_sentence.split(); each target token is mapped to the
         whitespace word it lies in;
  4. tags the English words with spaCy exactly as scripts/02_project_labels.py does
     (PERSON->PER, GPE/LOC->LOC, ORG->ORG).

Output: one JSONL per language with
  {"lang", "pair_id", "en_words", "tgt_words", "en_tags",
   "links": {"simalign": [[i, j], ...], "awesome": [...], "hybrid": [...]},
   "hybrid_scores": [...]}   (hybrid_scores[k] belongs to links["hybrid"][k])

Usage:
    python scripts/analysis/build_alignment_sample.py --n 200 --seed 42 \
        --out-dir results/alignment_eval
"""

import argparse
import glob
import gzip
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from naamasetu.languages import ALL_LANGS, lang_info  # noqa: E402

SPACY_LABEL_MAP = {"PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "ORG": "ORG"}
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def read_released(base):
    """Yield records from base.jsonl.gz or base.part1..N.jsonl.gz, in order."""
    parts = sorted(glob.glob(base + ".part*.jsonl.gz"),
                   key=lambda p: int(re.search(r"\.part(\d+)\.", p).group(1)))
    for p in parts or [base + ".jsonl.gz"]:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def first_key(d, *keys):
    for k in keys:
        if d.get(k):
            return d[k]
    return None


def baseline_index(lang, kind):
    """{(tgt_sentence, en_sentence): [(en_word, tgt_word), ...]} for simalign / awesome."""
    out = {}
    for d in read_released(os.path.join(ROOT, "data", "parallel", lang, f"{kind}_{lang}")):
        tgt = first_key(d, f"{lang}_sentence", "malayalam", "assamese_sentence")
        en = first_key(d, "en_sentence", "english", "english_sentence")
        links = []
        for a in d.get("alignments") or d.get("aligned_words") or []:
            ew = first_key(a, "en_word", "en")
            tw = first_key(a, f"{lang}_word", lang, "ml_word")
            if ew and tw:
                links.append((ew, tw))
        out[(tgt, en)] = links
    return out


def hybrid_index(lang):
    out = {}
    for d in read_released(os.path.join(ROOT, "data", "projected", lang, f"alignment_results_{lang}")):
        tgt = first_key(d, f"{lang}_sentence", "tgt_sentence")
        out[(tgt, d["en_sentence"])] = [(a["src_idx"], a["tgt_idx"], a["score"]) for a in d["alignments"]]
    return out


def tag_english(doc, en_words):
    tags = ["O"] * len(en_words)
    text = " ".join(en_words)
    offsets, cur = [], 0
    for w in en_words:
        s = text.index(w, cur); offsets.append((s, s + len(w))); cur = s + len(w)
    for ent in doc.ents:
        lab = SPACY_LABEL_MAP.get(ent.label_)
        if lab is None:
            continue
        hit = [i for i, (s, e) in enumerate(offsets) if s < ent.end_char and e > ent.start_char]
        for k, i in enumerate(hit):
            tags[i] = ("B-" if k == 0 else "I-") + lab
    return tags


def _core(w):
    return re.sub(r"[^\wऀ-෿]", "", w)


def _find(word, words, used):
    """First not-yet-linked occurrence of `word` (exact, else ignoring punctuation)."""
    # exact; then ignoring punctuation; then the word containing the token (the
    # Malayalam SimAlign run split "India's" into "India" + "'s")
    for same in (lambda w: w == word, lambda w: _core(w) == _core(word) and _core(word),
                 lambda w: word in w):
        hit = next((k for k, w in enumerate(words) if same(w) and k not in used),
                   next((k for k, w in enumerate(words) if same(w)), None))
        if hit is not None:
            return hit
    return None


def map_strings(links, en_words, tgt_words):
    """Word-string links -> whitespace-word index links (first free occurrence)."""
    used_e, used_t, out = set(), set(), []
    for ew, tw in links:
        i, j = _find(ew, en_words, used_e), _find(tw, tgt_words, used_t)
        if i is not None and j is not None:
            out.append([i, j]); used_e.add(i); used_t.add(j)
    return out


def token_to_word(tgt_sentence, tokens, tgt_words):
    """Map each Indic-NLP token to the whitespace word it lies in (by character search)."""
    starts, pos = [], 0
    for w in tgt_words:
        s = tgt_sentence.index(w, pos); starts.append((s, s + len(w))); pos = s + len(w)
    out, pos = [], 0
    for t in tokens:
        s = tgt_sentence.find(t, pos)
        if s < 0:
            out.append(None); continue
        pos = s + len(t)
        out.append(next((k for k, (a, b) in enumerate(starts) if a <= s < b), None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=ALL_LANGS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/alignment_eval")
    args = ap.parse_args()

    import spacy
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    nlp = spacy.load("en_core_web_sm")
    os.makedirs(args.out_dir, exist_ok=True)

    for lang in args.langs:
        code = lang_info(lang)["indic"]
        try:
            norm = IndicNormalizerFactory().get_normalizer(code)
        except Exception:
            norm = None
        sim, awe, hyb = baseline_index(lang, "simalign"), baseline_index(lang, "awesome"), hybrid_index(lang)
        common = sorted(set(sim) & set(awe) & set(hyb))
        order = common[:]
        random.Random(args.seed).shuffle(order)       # take pairs in this order until N are usable
        path = os.path.join(args.out_dir, f"sample_{lang}.jsonl")
        skipped = kept = 0
        lost = {"simalign": [0, 0], "awesome": [0, 0], "hybrid": [0, 0]}   # [unmapped, total]
        with open(path, "w", encoding="utf-8") as f:
            for tgt, en in order:
                if kept == args.n:
                    break
                en_words, tgt_words = en.split(), tgt.split()
                # the hybrid aligner tokenised the normalised sentence; map on that text
                tgt_norm = norm.normalize(tgt) if norm else tgt
                toks = indic_tokenize.trivial_tokenize(tgt_norm, lang=code)
                try:
                    t2w = token_to_word(tgt_norm, toks, tgt_norm.split())
                except ValueError:
                    skipped += 1; continue
                if len(tgt_norm.split()) != len(tgt_words):
                    skipped += 1; continue
                hl, hs = [], []
                for ti, ej, s in hyb[(tgt, en)]:          # swapped fields, see docstring
                    if ej < len(en_words) and ti < len(t2w) and t2w[ti] is not None:
                        hl.append([ej, t2w[ti]]); hs.append(s)
                links = {"simalign": map_strings(sim[(tgt, en)], en_words, tgt_words),
                         "awesome": map_strings(awe[(tgt, en)], en_words, tgt_words),
                         "hybrid": hl}
                for m, raw in (("simalign", sim[(tgt, en)]), ("awesome", awe[(tgt, en)]),
                               ("hybrid", hyb[(tgt, en)])):
                    lost[m][0] += len(raw) - len(links[m]); lost[m][1] += len(raw)
                rec = {"lang": lang, "pair_id": kept, "en_words": en_words, "tgt_words": tgt_words,
                       "en_tags": tag_english(nlp(en), en_words), "links": links,
                       "hybrid_scores": hs}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{lang}: {len(common):,} pairs aligned by all three; kept {kept} (skipped {skipped}); "
              "unmapped links " + ", ".join(f"{m} {a}/{b}" for m, (a, b) in lost.items()) + f" -> {path}")


if __name__ == "__main__":
    main()
