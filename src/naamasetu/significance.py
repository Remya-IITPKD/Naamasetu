"""Paired bootstrap significance test for NER F1 (span- or token-level).

Compares two systems (e.g. gold-only vs gold+silver) evaluated on the SAME
test sentences. Sentences are resampled with replacement; in each resample,
micro F1 is recomputed for both systems from per-sentence TP/FP/FN
counts (Koehn, 2004; Berg-Kirkpatrick et al., 2012). `metric="span"` counts
exact-match entities; `metric="token"` counts non-O tags (labels.token_prf);
`metric="type"` counts words after merging B-/I- (labels.type_token_prf).

Reported:
    delta      F1(system) - F1(baseline) on the full test set
    ci_low/hi  percentile 95% CI of delta over resamples
    p_value    one-sided: fraction of resamples with delta <= 0
"""

import numpy as np

from .labels import ENTITY_TYPES, extract_spans, to_type


def sentence_counts(gold_seqs, pred_seqs, entity_types=ENTITY_TYPES, metric="span"):
    """Per-sentence (tp, fp, fn) arrays for micro span or token F1."""
    if metric in ("token", "type"):
        tp, fp, fn = [], [], []
        for g, p in zip(gold_seqs, pred_seqs):
            if metric == "type":
                g, p = [to_type(x) for x in g], [to_type(x) for x in p]
            tp.append(sum(a == b != "O" for a, b in zip(g, p)))
            fp.append(sum(a != b and b != "O" for a, b in zip(g, p)))
            fn.append(sum(a != b and a != "O" for a, b in zip(g, p)))
        return np.array(tp), np.array(fp), np.array(fn)
    if metric != "span":
        raise ValueError(f"metric must be 'span', 'token' or 'type', not {metric!r}")
    types = set(entity_types)
    tp, fp, fn = [], [], []
    for g, p in zip(gold_seqs, pred_seqs):
        gs = {s for s in extract_spans(g) if s[2] in types}
        ps = {s for s in extract_spans(p) if s[2] in types}
        n = len(gs & ps)
        tp.append(n)
        fp.append(len(ps) - n)
        fn.append(len(gs) - n)
    return np.array(tp), np.array(fp), np.array(fn)


def _f1(tp, fp, fn):
    """Micro F1 from summed counts; works on scalars or arrays."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)


def paired_bootstrap(gold_seqs, base_pred, sys_pred, n_boot=10000, seed=12345,
                     alpha=0.05, chunk=500, metric="span"):
    """Paired bootstrap of F1(sys) - F1(base) over test sentences."""
    if not (len(gold_seqs) == len(base_pred) == len(sys_pred)):
        raise ValueError("gold / baseline / system must cover the same sentences")
    b = sentence_counts(gold_seqs, base_pred, metric=metric)
    s = sentence_counts(gold_seqs, sys_pred, metric=metric)
    f_base = float(_f1(*(x.sum() for x in b)))
    f_sys = float(_f1(*(x.sum() for x in s)))

    rng = np.random.default_rng(seed)
    n = len(gold_seqs)
    deltas = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        fb = _f1(*(x[idx].sum(axis=1) for x in b))
        fs = _f1(*(x[idx].sum(axis=1) for x in s))
        deltas.append(fs - fb)
    deltas = np.concatenate(deltas)

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_sentences": n,
        "n_boot": n_boot,
        "f1_baseline": f_base,
        "f1_system": f_sys,
        "delta": f_sys - f_base,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(np.mean(deltas <= 0)),
    }
