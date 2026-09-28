"""Label schema and span-level NER metrics shared by training and evaluation.

Span metrics follow the CoNLL convention: an entity is correct only if both its
boundaries and its type match exactly. An I- tag that does not continue a span
of the same type starts a new span (IOB1-style repair), matching seqeval's
default (non-strict) behaviour.
"""

from collections import defaultdict

TAG_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
TAG2ID = {t: i for i, t in enumerate(TAG_LIST)}
ID2TAG = {i: t for t, i in TAG2ID.items()}
ENTITY_TYPES = ["PER", "LOC", "ORG"]

# Naamapadam releases integer tags in this order (MISC ids never occur in the
# PER/LOC/ORG subset but are kept so that legacy files still decode).
NAAMAPADAM_INT_LABELS = TAG_LIST + ["B-MISC", "I-MISC"]


def decode_tags(tags):
    """Accept either string tags or Naamapadam integer ids."""
    if tags and isinstance(tags[0], int):
        return [NAAMAPADAM_INT_LABELS[i] if i < len(NAAMAPADAM_INT_LABELS) else "O"
                for i in tags]
    return list(tags)


def extract_spans(tag_seq):
    """Return a set of (start, end_exclusive, type) spans from a BIO sequence."""
    spans = set()
    start, cur = None, None
    for i, tag in enumerate(tag_seq):
        if tag.startswith("B-"):
            if cur is not None:
                spans.add((start, i, cur))
            start, cur = i, tag[2:]
        elif tag.startswith("I-"):
            etype = tag[2:]
            if cur != etype:
                if cur is not None:
                    spans.add((start, i, cur))
                start, cur = i, etype
        else:
            if cur is not None:
                spans.add((start, i, cur))
            start = cur = None
    if cur is not None:
        spans.add((start, len(tag_seq), cur))
    return spans


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def token_prf(y_true_seqs, y_pred_seqs):
    """Token-level micro P/R/F1 over the non-O tags (first sub-token per word).

    Equivalent to sklearn `precision_recall_fscore_support(average="micro",
    labels=<all tags except O>)` on the flattened sequences. More lenient than
    span F1: a partly correct entity still earns credit for its correct tokens.
    """
    tp = fp = fn = 0
    for t_seq, p_seq in zip(y_true_seqs, y_pred_seqs):
        for t, p in zip(t_seq, p_seq):
            if p == t:
                tp += t != "O"
            else:
                fp += p != "O"
                fn += t != "O"
    return _prf(tp, fp, fn)


def to_type(tag):
    """B-PER / I-PER -> PER; O -> O."""
    return tag.split("-", 1)[1] if "-" in tag else "O"


def type_token_prf(y_true_seqs, y_pred_seqs, entity_types=ENTITY_TYPES):
    """Token-level micro P/R/F1 after merging B-/I- into the entity type.

    Each word counts as PER, LOC, ORG or O; micro-averaged over the entity
    types (O excluded). Returns (micro_p, micro_r, micro_f1), {type: (p, r, f1, support)}.
    Most lenient of the three metrics here: a word is right if its type is
    right, whatever its B/I position.
    """
    tp = {e: 0 for e in entity_types}; fp = dict(tp); fn = dict(tp); support = dict(tp)
    for t_seq, p_seq in zip(y_true_seqs, y_pred_seqs):
        for t, p in zip(t_seq, p_seq):
            t, p = to_type(t), to_type(p)
            if t in support:
                support[t] += 1
            if t == p:
                if t in tp:
                    tp[t] += 1
            else:
                if p in fp:
                    fp[p] += 1
                if t in fn:
                    fn[t] += 1
    micro = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_type = {e: (*_prf(tp[e], fp[e], fn[e]), support[e]) for e in entity_types}
    return micro, per_type


def span_prf(y_true_seqs, y_pred_seqs, entity_types=ENTITY_TYPES):
    """Span-level micro P/R/F1 plus per-type scores.

    Returns (micro_p, micro_r, micro_f1), {type: (p, r, f1, support)}, report_str
    """
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    support = defaultdict(int)
    for t_seq, p_seq in zip(y_true_seqs, y_pred_seqs):
        t_sp, p_sp = extract_spans(t_seq), extract_spans(p_seq)
        for s in t_sp:
            support[s[2]] += 1
            if s not in p_sp:
                fn[s[2]] += 1
        for s in p_sp:
            (tp if s in t_sp else fp)[s[2]] += 1

    micro = _prf(sum(tp[e] for e in entity_types),
                 sum(fp[e] for e in entity_types),
                 sum(fn[e] for e in entity_types))
    per_type = {e: (*_prf(tp[e], fp[e], fn[e]), support[e]) for e in entity_types}

    total = sum(support[e] for e in entity_types)
    lines = [f"{'':>12}{'precision':>12}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for e in entity_types:
        p, r, f, s = per_type[e]
        lines.append(f"{e:>12}{p:>12.4f}{r:>10.4f}{f:>10.4f}{s:>10}")
    lines.append("")
    lines.append(f"{'micro avg':>12}{micro[0]:>12.4f}{micro[1]:>10.4f}{micro[2]:>10.4f}{total:>10}")
    macro = [sum(per_type[e][k] for e in entity_types) / len(entity_types) for k in range(3)]
    lines.append(f"{'macro avg':>12}{macro[0]:>12.4f}{macro[1]:>10.4f}{macro[2]:>10.4f}{total:>10}")
    return micro, per_type, "\n".join(lines)
