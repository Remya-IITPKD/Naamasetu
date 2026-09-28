import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.labels import span_prf  # noqa: E402
from naamasetu.significance import paired_bootstrap, sentence_counts  # noqa: E402

GOLD = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O"], ["O", "B-LOC"]] * 20


def test_counts_match_span_prf():
    pred = [["B-PER", "O", "O", "B-LOC"], ["B-ORG", "O"], ["O", "O"]] * 20
    tp, fp, fn = sentence_counts(GOLD, pred)
    (p, r, f), _, _ = span_prf(GOLD, pred)
    assert abs(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()) - f) < 1e-12


def test_identical_systems_not_significant():
    r = paired_bootstrap(GOLD, GOLD, GOLD, n_boot=200)
    assert r["delta"] == 0 and r["p_value"] == 1.0


def test_clear_improvement_is_significant():
    weak = [["O", "O", "O", "B-LOC"], ["O", "O"], ["O", "O"]] * 20
    r = paired_bootstrap(GOLD, weak, GOLD, n_boot=500)
    assert r["delta"] > 0 and r["ci_low"] > 0 and r["p_value"] < 0.01


def test_deterministic_given_seed():
    weak = [["O", "O", "O", "B-LOC"], ["B-ORG", "O"], ["O", "B-LOC"]] * 20
    a = paired_bootstrap(GOLD, weak, GOLD, n_boot=300, seed=7)
    b = paired_bootstrap(GOLD, weak, GOLD, n_boot=300, seed=7)
    assert a == b


def test_token_counts_match_token_prf():
    from naamasetu.labels import token_prf
    pred = [["B-PER", "O", "B-ORG", "B-LOC"], ["B-ORG", "O"], ["O", "O"]] * 20
    tp, fp, fn = sentence_counts(GOLD, pred, metric="token")
    f = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    assert abs(f - token_prf(GOLD, pred)[2]) < 1e-12


def test_type_counts_match_type_token_prf():
    from naamasetu.labels import type_token_prf
    pred = [["I-PER", "B-PER", "B-ORG", "B-LOC"], ["B-ORG", "O"], ["O", "O"]] * 20
    tp, fp, fn = sentence_counts(GOLD, pred, metric="type")
    f = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    assert abs(f - type_token_prf(GOLD, pred)[0][2]) < 1e-12
