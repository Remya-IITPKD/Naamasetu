import importlib.util
import os

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "proj", os.path.join(HERE, "..", "scripts", "02_project_labels.py"))
proj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proj)


def test_project_bio_rederivation():
    scored = [{"src_idx": 0, "tgt_idx": 1, "score": 0.9},
              {"src_idx": 1, "tgt_idx": 2, "score": 0.8}]
    src = ["B-PER", "I-PER", "O"]
    tgt = ["x", "a", "b"]
    assert proj.project_tags(scored, src, 3, tgt) == ["O", "B-PER", "I-PER"]


def test_highest_score_wins_target():
    scored = [{"src_idx": 0, "tgt_idx": 0, "score": 0.5},
              {"src_idx": 1, "tgt_idx": 0, "score": 0.9}]
    assert proj.project_tags(scored, ["B-PER", "B-LOC"], 1, ["a"]) == ["B-LOC"]


def test_punctuation_not_labelled():
    scored = [{"src_idx": 0, "tgt_idx": 0, "score": 0.9}]
    assert proj.project_tags(scored, ["B-ORG"], 1, ["."]) == ["O"]


def test_mutual_nn():
    import numpy as np
    H = np.array([[0.9, 0.1], [0.8, 0.2]])
    assert proj.HybridAligner.mutual_nn(H) == [(0, 0)]
