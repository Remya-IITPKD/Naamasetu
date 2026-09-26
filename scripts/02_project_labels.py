#!/usr/bin/env python3
"""Stage 2 — Hybrid word alignment and NER label projection (English -> Indic).

Implements Algorithm 1 of the paper:

    Require: source s (English, labels L_S from spaCy), target t (Indic),
             encoder E, weights a,b,g, entity bonus L, phonetic threshold th,
             similarity floor tau, Itermax iterations K
      U <- E(s); V <- E(t)                              # contextual word vectors
      for all (i,j):
        Sem[i,j]   = cos(U_i, V_j)                      # meaning
        Pho[i,j]   = 1 - ed(ph(s_i), ph(t_j))           # sound  (Metaphone)
        Roman[i,j] = 1 - ed(rm(s_i), rm(t_j))           # spelling (ITRANS)
        H[i,j] = 1                if Pho > th            # strong phonetic match
               = a*Sem+b*Pho+g*Roman  otherwise
        if L_S[i] != O: H[i,j] += L                     # entity bonus
      H <- H / max(H)
      A <- mutual nearest neighbours of H
      repeat K times (Itermax): add mutual NN among still-unaligned rows/cols
                                with H >= tau
      A <- {(i,j) in A : H[i,j] >= tau}
      project labels over A (highest-score pair wins each target token),
      unaligned targets = O, re-derive valid BIO.

Usage:
    python scripts/02_project_labels.py --lang mr \
        --input data/parallel/mr/parallel_mr_en.jsonl \
        --output-dir data/projected/mr \
        --model-path bert-base-multilingual-cased

Input  : JSONL with `en_sentence` and a target sentence under one of
         `<lang>_sentence`, `tgt_sentence`, `target_sentence`.
Outputs (in --output-dir):
    alignment_results_<lang>.jsonl   word alignments with scores
    english_ner_tags_<lang>.jsonl    spaCy source labels
    <lang>_ner_tags.jsonl            projected target labels
    silver_projected_<lang>.jsonl    {id, tokens, ner_tags} (training format)
Resumable: restarts from the last complete line if interrupted.
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamsetu.languages import ALL_LANGS, lang_info, script_checker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPACY_LABEL_MAP = {"PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "ORG": "ORG"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=ALL_LANGS)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="bert-base-multilingual-cased",
                   help="HF id or local path of the encoder E (mBERT in the paper)")
    p.add_argument("--offline", action="store_true",
                   help="load encoder with local_files_only=True (for air-gapped clusters)")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--layer", type=int, default=8, help="hidden layer used for word vectors")
    # fusion weights and thresholds (paper defaults)
    p.add_argument("--alpha", type=float, default=0.6, help="a: weight on Sem")
    p.add_argument("--beta", type=float, default=0.2, help="b: weight on Pho")
    p.add_argument("--gamma", type=float, default=0.2, help="g: weight on Roman")
    p.add_argument("--lam", type=float, default=0.15, help="L: entity bonus")
    p.add_argument("--theta", type=float, default=0.8, help="th: phonetic hard-match threshold")
    p.add_argument("--tau", type=float, default=0.40, help="tau: similarity floor on normalised H")
    p.add_argument("--iters", type=int, default=2, help="K: Itermax iterations")
    p.add_argument("--phonetic-min-len", type=int, default=4,
                   help="min romanised length before Pho/Roman are trusted "
                        "(guards short function words against Metaphone collisions)")
    p.add_argument("--max-pairs", type=int, default=None, help="debug: stop after N pairs")
    return p.parse_args()


# ---------------------------------------------------------------- romanisation
_ALPHA_RE = re.compile(r"[^a-z]")


def make_roman_fns(lang):
    import jellyfish
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    scheme = getattr(sanscript, lang_info(lang)["scheme"])

    def roman_en(w):
        return _ALPHA_RE.sub("", w.lower())

    def roman_tgt(w):
        try:
            r = transliterate(w, scheme, sanscript.ITRANS)
        except Exception:
            r = w
        return _ALPHA_RE.sub("", r.lower())

    def metaphone(s):
        try:
            return jellyfish.metaphone(s) if s else ""
        except Exception:
            return ""

    def norm_edit_sim(a, b):
        if not a or not b:
            return 0.0
        return 1.0 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b))

    return roman_en, roman_tgt, metaphone, norm_edit_sim


# ---------------------------------------------------------------- encoder
class WordEmbedder:
    """Mean-pools sub-word states of one hidden layer into word vectors."""

    def __init__(self, model_path, device, layer, offline):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device if (torch.cuda.is_available() and device != "cpu") else "cpu")
        logger.info(f"Loading encoder on {self.device} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, output_hidden_states=True,
                                         local_files_only=offline)
        if layer >= cfg.num_hidden_layers + 1:
            raise ValueError(f"layer {layer} out of range (0-{cfg.num_hidden_layers})")
        self.model = AutoModel.from_pretrained(model_path, config=cfg,
                                               local_files_only=offline).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=offline)
        self.layer, self.dim = layer, cfg.hidden_size

    def embed(self, words):
        if not words:
            return np.empty((0, self.dim)), []
        with self.torch.no_grad():
            inp = self.tok(words, is_split_into_words=True, return_tensors="pt",
                           truncation=True, padding=True).to(self.device)
            out = self.model(**inp)["hidden_states"][self.layer][0]
        wids = inp.word_ids(batch_index=0)
        kept = sorted({w for w in wids if w is not None})  # words surviving truncation
        vecs = [out[[k for k, w in enumerate(wids) if w == wi]].mean(dim=0).cpu().numpy()
                for wi in kept]
        return np.array(vecs), kept


# ---------------------------------------------------------------- aligner
class HybridAligner:
    def __init__(self, embedder, roman_fns, a, b, g, lam, theta, tau, iters, min_len):
        self.E = embedder
        self.roman_en, self.roman_tgt, self.metaphone, self.ned = roman_fns
        self.a, self.b, self.g = a, b, g
        self.lam, self.theta, self.tau, self.iters, self.min_len = lam, theta, tau, iters, min_len

    @staticmethod
    def _sem(U, V):
        from sklearn.metrics.pairwise import cosine_similarity
        if U.size == 0 or V.size == 0:
            return np.zeros((U.shape[0], V.shape[0]))
        return (cosine_similarity(U, V) + 1.0) / 2.0  # -> [0,1]

    def build_H(self, U, V, src_words, tgt_words, src_labels):
        n, m = U.shape[0], V.shape[0]
        Sem = self._sem(U, V)
        H = np.zeros((n, m))
        rm_s = [self.roman_en(w) for w in src_words]
        rm_t = [self.roman_tgt(w) for w in tgt_words]
        ph_s = [self.metaphone(r) for r in rm_s]
        ph_t = [self.metaphone(r) for r in rm_t]
        for i in range(n):
            is_ent = i < len(src_labels) and src_labels[i] != "O"
            si_ok = len(rm_s[i]) >= self.min_len
            for j in range(m):
                trust = si_ok and len(rm_t[j]) >= self.min_len
                pho = self.ned(ph_s[i], ph_t[j]) if trust else 0.0
                rom = self.ned(rm_s[i], rm_t[j]) if trust else 0.0
                h = 1.0 if (trust and pho > self.theta) else \
                    self.a * Sem[i, j] + self.b * pho + self.g * rom
                if is_ent:
                    h += self.lam
                H[i, j] = h
        mx = H.max() if H.size else 0.0
        return (H / mx if mx > 0 else H), Sem

    @staticmethod
    def mutual_nn(H, rows=None, cols=None):
        n, m = H.shape
        if n == 0 or m == 0:
            return []
        W = H.copy()
        if rows is not None:
            mask = np.ones(n, bool); mask[list(rows)] = False; W[mask, :] = -np.inf
        if cols is not None:
            mask = np.ones(m, bool); mask[list(cols)] = False; W[:, mask] = -np.inf
        if not np.isfinite(W).any():
            return []
        fwd, bwd = np.argmax(W, axis=1), np.argmax(W, axis=0)
        return [(i, int(fwd[i])) for i in range(n)
                if np.isfinite(W[i, fwd[i]]) and int(bwd[fwd[i]]) == i]

    def align(self, src_words, tgt_words, src_labels):
        if not src_words or not tgt_words:
            return [], src_words, tgt_words, [], list(range(len(tgt_words)))
        U, s_keep = self.E.embed(src_words)
        V, t_keep = self.E.embed(tgt_words)
        if U.size == 0 or V.size == 0:
            return [], src_words, tgt_words, list(range(len(src_words))), list(range(len(tgt_words)))
        src_used = [src_words[i] for i in s_keep]
        tgt_used = [tgt_words[j] for j in t_keep]
        lab_used = [src_labels[i] if i < len(src_labels) else "O" for i in s_keep]

        H, _ = self.build_H(U, V, src_used, tgt_used, lab_used)
        A = set(self.mutual_nn(H))
        for _ in range(self.iters):  # Itermax on residual words
            ur, uc = {i for i, _ in A}, {j for _, j in A}
            rr = [i for i in range(H.shape[0]) if i not in ur]
            rc = [j for j in range(H.shape[1]) if j not in uc]
            if not rr or not rc:
                break
            new = [(i, j) for i, j in self.mutual_nn(H, rr, rc) if H[i, j] >= self.tau and (i, j) not in A]
            if not new:
                break
            A.update(new)
        A = [(i, j) for i, j in A if H[i, j] >= self.tau]
        scored = sorted(({"src_idx": i, "tgt_idx": j, "score": round(float(H[i, j]), 4)}
                         for i, j in A), key=lambda x: x["score"], reverse=True)
        return scored, src_used, tgt_used, s_keep, t_keep


# ---------------------------------------------------------------- labelling
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


def project_tags(scored, src_labels, n_tgt, tgt_words):
    """Highest-scoring pair wins each target token; then re-derive BIO."""
    base = ["O"] * n_tgt
    for pr in sorted(scored, key=lambda x: x["score"], reverse=True):
        i, j = pr["src_idx"], pr["tgt_idx"]
        if j >= n_tgt or i >= len(src_labels):
            continue
        tag, tw = src_labels[i], (tgt_words[j] if j < len(tgt_words) else "")
        if tag != "O" and any(c.isalnum() for c in tw) and base[j] == "O":
            base[j] = tag.split("-")[-1]
    return [("O" if b == "O" else (f"B-{b}" if k == 0 or base[k - 1] != b else f"I-{b}"))
            for k, b in enumerate(base)]


def recover_checkpoint(path):
    if not os.path.exists(path):
        return 0
    good = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line); good.append(line)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt line at {len(good)} -- truncating.")
                break
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(l + "\n" for l in good)
    return len(good)


def main():
    args = parse_args()
    import spacy
    import torch
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize

    info = lang_info(args.lang)
    has_tgt_script = script_checker(args.lang)
    try:
        normalizer = IndicNormalizerFactory().get_normalizer(info["indic"])
    except Exception:
        normalizer = None

    def tok_tgt(s) -> List[str]:
        return indic_tokenize.trivial_tokenize(normalizer.normalize(s) if normalizer else s,
                                               lang=info["indic"])

    def is_english(s):
        return not has_tgt_script(s) and bool(re.search(r"[a-zA-Z]", s))

    def get_tgt(item):
        for k in (f"{args.lang}_sentence", "tgt_sentence", "target_sentence"):
            if item.get(k):
                return item[k]
        return ""

    os.makedirs(args.output_dir, exist_ok=True)
    L = args.lang
    out = {k: os.path.join(args.output_dir, v) for k, v in {
        "align": f"alignment_results_{L}.jsonl",
        "en": f"english_ner_tags_{L}.jsonl",
        "tgt": f"{L}_ner_tags.jsonl",
        "silver": f"silver_projected_{L}.jsonl"}.items()}

    pairs = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if is_english(item.get("en_sentence", "")):
                    pairs.append(item)
                    if args.max_pairs and len(pairs) >= args.max_pairs:
                        break
    logger.info(f"{info['name']}: loaded {len(pairs)} English-{info['name']} pairs")

    aligner = HybridAligner(
        WordEmbedder(args.model_path, "cuda" if torch.cuda.is_available() else "cpu",
                     args.layer, args.offline),
        make_roman_fns(L), args.alpha, args.beta, args.gamma, args.lam,
        args.theta, args.tau, args.iters, args.phonetic_min_len)
    nlp = spacy.load(args.spacy_model)

    start = recover_checkpoint(out["silver"])
    if start:
        for k in ("align", "en", "tgt"):
            recover_checkpoint(out[k])
    logger.info(f"Resuming from index {start}")

    fh = {k: open(v, "a", encoding="utf-8") for k, v in out.items()}
    try:
        for idx in range(start, len(pairs)):
            p = pairs[idx]
            tgt_sent, en_sent = get_tgt(p), p.get("en_sentence", "")
            if not tgt_sent:
                logger.warning(f"Missing target sentence at {idx}; skipping.")
                continue
            en_words, tgt_words = en_sent.split(), tok_tgt(tgt_sent)
            en_tags = tag_english(nlp(en_sent), en_words)
            scored, src_used, tgt_used, s_keep, _ = aligner.align(en_words, tgt_words, en_tags)
            src_labels = [en_tags[i] if i < len(en_tags) else "O" for i in s_keep]
            tgt_tags = project_tags(scored, src_labels, len(tgt_used), tgt_used)

            dump = lambda k, obj: fh[k].write(json.dumps(obj, ensure_ascii=False) + "\n")
            dump("align", {"en_sentence": en_sent, "tgt_sentence": tgt_sent, "alignments": scored})
            dump("en", {"sentence": en_sent, "tokens": src_used, "tags": src_labels})
            dump("tgt", {"sentence": tgt_sent, "tokens": tgt_used, "tags": tgt_tags})
            dump("silver", {"id": f"{L}_{idx}", "tokens": tgt_used, "ner_tags": tgt_tags})

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(pairs)}")
                for h in fh.values():
                    h.flush(); os.fsync(h.fileno())
    finally:
        for h in fh.values():
            h.close()
    logger.info("Projection complete.")


if __name__ == "__main__":
    main()
