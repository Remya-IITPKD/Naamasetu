import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.labels import decode_tags, extract_spans, span_prf, token_prf  # noqa: E402


def test_extract_simple():
    assert extract_spans(["B-PER", "I-PER", "O", "B-LOC"]) == {(0, 2, "PER"), (3, 4, "LOC")}


def test_orphan_I_starts_span():
    assert extract_spans(["O", "I-ORG", "I-ORG"]) == {(1, 3, "ORG")}


def test_type_switch_splits():
    assert extract_spans(["B-PER", "I-LOC"]) == {(0, 1, "PER"), (1, 2, "LOC")}


def test_perfect_and_partial():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    (p, r, f), _, _ = span_prf(gold, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)
    pred = [["B-PER", "O", "O", "B-LOC"]]            # PER boundary wrong
    (p, r, f), per, _ = span_prf(gold, pred)
    assert p == 0.5 and r == 0.5 and per["LOC"][2] == 1.0 and per["PER"][2] == 0.0


def test_token_prf_counts_partial_entities():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    pred = [["B-PER", "O", "B-ORG", "B-LOC"]]
    # tp: B-PER, B-LOC; fp: B-ORG; fn: I-PER
    p, r, f = token_prf(gold, pred)
    assert (p, r) == (2 / 3, 2 / 3) and abs(f - 2 / 3) < 1e-12
    (sp, sr, sf), _, _ = span_prf(gold, pred)
    assert sf < f                                    # span F1 is stricter


def test_decode_int_tags():
    assert decode_tags([0, 1, 2, 5]) == ["O", "B-PER", "I-PER", "B-LOC"]
