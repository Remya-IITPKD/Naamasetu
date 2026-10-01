#!/usr/bin/env python3
"""Entity-centric hybrid alignment and projection: Algorithm 1 of the paper,
implemented line by line.

    1: U <- E(s); V <- E(t)                       contextual embeddings
    2: for all source-target pairs (i, j):
    3:     Sem[i,j]   <- cos(U_i, V_j)            meaning
    4:     Pho[i,j]   <- 1 - ed(ph(s_i), ph(t_j)) sound
    5:     Roman[i,j] <- 1 - ed(rm(s_i), rm(t_j)) spelling
    6-10:  H[i,j] <- 1 if Pho[i,j] > theta else alpha*Sem + beta*Pho + gamma*Roman
    11:    if L_S[i] != O: H[i,j] += lambda      entity bonus
    14: H <- normalize(H)                         divide by the maximum of H
    15-18: Argmax both ways; Match = mutual nearest neighbours
    19-23: Itermax: K rounds on the unaligned rows/columns, adding new mutual
           pairs with H >= tau
    24: similarity floor: keep (i, j) with H[i,j] >= tau
    25-27: project source labels (1-1, 1-many, many-1)
    28: unaligned targets <- O; validate BIO

Definitions used for the symbols of the algorithm:
  E      mBERT (bert-base-multilingual-cased), layer 8; a word vector is the mean of
         its sub-word states
  rm(.)  romanisation: Indic words are transliterated to ITRANS; both sides are
         lower-cased and reduced to the letters a-z
  ph(.)  Metaphone code of the romanised word (jellyfish.metaphone)
  ed     Levenshtein distance divided by the length of the longer string
  s      English words (whitespace split); t: Indic words (Indic NLP tokeniser
         after Unicode normalisation)
  L_S    English BIO labels from spaCy (PERSON->PER, GPE/LOC->LOC, ORG->ORG)

Projection (lines 25-27): a target word linked to several source words takes the
label of the entity source word with the highest H (many-1); consecutive target
words linked to the same English entity become B-, I-, ... (1-many); otherwise the
target word starts a new entity (1-1).

Output (one line per pair, same order as the input):
  alignment_results_<lang>.jsonl   {"en_sentence", "tgt_sentence",
                                    "alignments": [{"src_idx", "tgt_idx", "score"}]}
                                   src_idx indexes the English words, tgt_idx the
                                   Indic words, score = H[i, j]
  silver_projected_<lang>.jsonl    {"id", "tokens", "ner_tags"}

Usage:
    python scripts/naamasetu_align.py --lang ml \
        --input data/parallel/ml/parallel_ml_en.jsonl --output-dir data/projected/ml_alg1
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.languages import ALL_LANGS, lang_info, script_checker  # noqa: E402

SPACY_LABEL_MAP = {"PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "ORG": "ORG"}
_NON_LETTERS = re.compile(r"[^a-z]")


# ------------------------------------------------------------------ string signals
def norm_edit_distance(a: str, b: str) -> float:
    """ed(a, b) in [0, 1]: Levenshtein distance / length of the longer string."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)


def make_string_fns(lang: str):
    """Return rm_en, rm_tgt and ph (romanisation and phonetic coding)."""
    import jellyfish
    from indic_transliteration import sanscript
    scheme = getattr(sanscript, lang_info(lang)["scheme"])

    def rm_en(w):
        return _NON_LETTERS.sub("", w.lower())

    def rm_tgt(w):
        try:
            r = sanscript.transliterate(w, scheme, sanscript.ITRANS)
        except Exception:
            r = w
        return _NON_LETTERS.sub("", r.lower())

    def ph(w):
        return jellyfish.metaphone(w) if w else ""

    return rm_en, rm_tgt, ph


# ------------------------------------------------------------------ encoder E
class WordEncoder:
    """E(.): word vectors = mean of the word's sub-word states at one layer."""

    def __init__(self, model_path="bert-base-multilingual-cased", layer=8, offline=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.layer = torch, layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=offline)
        self.model = AutoModel.from_pretrained(model_path, output_hidden_states=True,
                                               local_files_only=offline).to(self.device).eval()

    def __call__(self, words: Sequence[str]) -> np.ndarray:
        dim = self.model.config.hidden_size
        out = np.zeros((len(words), dim), dtype=np.float32)   # words cut by truncation stay 0
        if not words:
            return out
        with self.torch.no_grad():
            enc = self.tok(list(words), is_split_into_words=True, return_tensors="pt",
                           truncation=True).to(self.device)
            hs = self.model(**enc).hidden_states[self.layer][0]
        wids = enc.word_ids(0)
        for w in {x for x in wids if x is not None}:
            out[w] = hs[[k for k, x in enumerate(wids) if x == w]].mean(0).cpu().numpy()
        return out


def cosine(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    un = np.linalg.norm(U, axis=1, keepdims=True)
    vn = np.linalg.norm(V, axis=1, keepdims=True)
    return (U @ V.T) / np.maximum(un * vn.T, 1e-12)


# ------------------------------------------------------------------ Algorithm 1
def hybrid_scores(Sem, Pho, Roman, src_labels, alpha, beta, gamma, lam, theta):
    """Lines 6-14: piecewise hybrid score, entity bonus, normalisation."""
    H = np.where(Pho > theta, 1.0, alpha * Sem + beta * Pho + gamma * Roman)   # 6-10
    for i, lab in enumerate(src_labels):                                        # 11
        if lab != "O":
            H[i, :] += lam
    mx = H.max() if H.size else 0.0                                            # 14
    return H / mx if mx > 0 else H


def _mutual(H: np.ndarray, rows: Set[int], cols: Set[int]) -> Set[Tuple[int, int]]:
    """Mutual nearest neighbours of H restricted to the given rows and columns."""
    if not rows or not cols:
        return set()
    r, c = sorted(rows), sorted(cols)
    sub = H[np.ix_(r, c)]
    fwd, bwd = sub.argmax(axis=1), sub.argmax(axis=0)
    return {(r[a], c[int(fwd[a])]) for a in range(len(r)) if int(bwd[fwd[a]]) == a}


def match_itermax(H: np.ndarray, tau: float, K: int) -> Set[Tuple[int, int]]:
    """Lines 15-24: Argmax, Match, K Itermax rounds, similarity floor."""
    m, n = H.shape
    A = _mutual(H, set(range(m)), set(range(n)))                               # 15-18
    for _ in range(K):                                                         # 20
        rows = set(range(m)) - {i for i, _ in A}                               # 21
        cols = set(range(n)) - {j for _, j in A}
        new = {(i, j) for i, j in _mutual(H, rows, cols) if H[i, j] >= tau}     # 22
        if not new:
            break
        A |= new
    return {(i, j) for i, j in A if H[i, j] >= tau}                            # 24


def entity_spans(labels: Sequence[str]) -> List[int]:
    """Span id of every source word (-1 for O); a span starts at B- or at a stray I-."""
    ids, cur, prev_type = [], -1, None
    for lab in labels:
        if lab == "O":
            ids.append(-1); prev_type = None; continue
        typ = lab[2:]
        if lab.startswith("B-") or typ != prev_type:
            cur += 1
        ids.append(cur); prev_type = typ
    return ids


def validate_bio(tags: List[str]) -> List[str]:
    """Line 28: an I-X that does not follow B-X/I-X becomes B-X."""
    out = []
    for t in tags:
        if t.startswith("I-") and (not out or out[-1][2:] != t[2:] or out[-1] == "O"):
            t = "B-" + t[2:]
        out.append(t)
    return out


def project(A: Set[Tuple[int, int]], H: np.ndarray, src_labels: Sequence[str],
            n_tgt: int) -> List[str]:
    """Lines 25-28: 1-1 / 1-many (B-, I-) / many-1 projection, then BIO validation."""
    span = entity_spans(src_labels)
    best: Dict[int, int] = {}                       # target j -> best entity source i
    for i, j in A:
        if src_labels[i] != "O" and (j not in best or H[i, j] > H[best[j], j]):
            best[j] = i                             # many-1: highest H wins
    tags = ["O"] * n_tgt                            # 28: unaligned targets -> O
    for j in range(n_tgt):
        if j not in best:
            continue
        i = best[j]
        typ = src_labels[i][2:]
        same_entity = j - 1 in best and span[best[j - 1]] == span[i]
        tags[j] = ("I-" if same_entity else "B-") + typ      # 1-many: B-, I-, ...
    return validate_bio(tags)


def align_and_project(src_words, tgt_words, src_labels, encoder, string_fns, p):
    """Algorithm 1 for one sentence pair. Returns (sorted links, H, target labels)."""
    rm_en, rm_tgt, ph = string_fns
    U, V = encoder(src_words), encoder(tgt_words)                              # 1
    Sem = cosine(U, V)                                                         # 3
    rs, rt = [rm_en(w) for w in src_words], [rm_tgt(w) for w in tgt_words]
    ps, pt = [ph(w) for w in rs], [ph(w) for w in rt]
    m, n = len(src_words), len(tgt_words)
    Pho, Roman = np.zeros((m, n)), np.zeros((m, n))
    for i in range(m):                                                         # 2
        for j in range(n):
            Pho[i, j] = 1.0 - norm_edit_distance(ps[i], pt[j])                 # 4
            Roman[i, j] = 1.0 - norm_edit_distance(rs[i], rt[j])               # 5
    H = hybrid_scores(Sem, Pho, Roman, src_labels, p.alpha, p.beta, p.gamma, p.lam, p.theta)
    A = match_itermax(H, p.tau, p.K)
    return sorted(A), H, project(A, H, src_labels, n)


# ------------------------------------------------------------------ English labels
def tag_english(doc, en_words: List[str]) -> List[str]:
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


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=ALL_LANGS)
    ap.add_argument("--input", required=True, help="parallel pairs JSONL")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="bert-base-multilingual-cased")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.15, help="entity bonus lambda")
    ap.add_argument("--theta", type=float, default=0.8, help="phonetic threshold")
    ap.add_argument("--tau", type=float, default=0.40, help="similarity floor")
    ap.add_argument("--K", type=int, default=2, help="Itermax iterations")
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    import spacy
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize

    info = lang_info(args.lang)
    try:
        normalizer = IndicNormalizerFactory().get_normalizer(info["indic"])
    except Exception:
        normalizer = None
    has_tgt_script = script_checker(args.lang)
    nlp = spacy.load(args.spacy_model)
    encoder = WordEncoder(args.model_path, args.layer, args.offline)
    fns = make_string_fns(args.lang)

    def tgt_text(d):
        return (d.get(f"{args.lang}_sentence") or d.get("tgt_sentence")
                or d.get("target_sentence") or "")

    os.makedirs(args.output_dir, exist_ok=True)
    out_align = open(os.path.join(args.output_dir, f"alignment_results_{args.lang}.jsonl"),
                     "w", encoding="utf-8")
    out_silver = open(os.path.join(args.output_dir, f"silver_projected_{args.lang}.jsonl"),
                      "w", encoding="utf-8")
    n = 0
    with open(args.input, encoding="utf-8") as f, out_align, out_silver:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            en, tgt = d.get("en_sentence", ""), tgt_text(d)
            if not tgt or has_tgt_script(en) or not re.search(r"[A-Za-z]", en):
                continue
            src_words = en.split()
            tgt_words = indic_tokenize.trivial_tokenize(
                normalizer.normalize(tgt) if normalizer else tgt, lang=info["indic"])
            if not src_words or not tgt_words:
                continue
            labels = tag_english(nlp(en), src_words)
            A, H, tgt_tags = align_and_project(src_words, tgt_words, labels, encoder, fns, args)
            out_align.write(json.dumps({
                "en_sentence": en, "tgt_sentence": tgt,
                "alignments": [{"src_idx": i, "tgt_idx": j, "score": round(float(H[i, j]), 4)}
                               for i, j in A]}, ensure_ascii=False) + "\n")
            out_silver.write(json.dumps({"id": f"{args.lang}_{n}", "tokens": tgt_words,
                                         "ner_tags": tgt_tags}, ensure_ascii=False) + "\n")
            n += 1
            if args.max_pairs and n >= args.max_pairs:
                break
    print(f"{args.lang}: aligned and projected {n} pairs -> {args.output_dir}")


if __name__ == "__main__":
    main()
