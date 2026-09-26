# Reproducibility notes

This file collects what the ARR Responsible NLP Checklist asks about
(sections B–C) so reviewers can find it in one place. Items marked TODO
must be filled before submission.

## Models

| Model | HF id | Parameters |
|---|---|---|
| mBERT | `bert-base-multilingual-cased` | ~178M |
| XLM-R base | `xlm-roberta-base` | ~278M |
| Sentence encoder (Stage 1) | `paraphrase-multilingual-mpnet-base-v2` | ~278M |
| English NER (Stage 2) | spaCy `en_core_web_sm` (TODO: version) | – |

## Hyperparameters

All values are in `configs/hyperparameters.yaml`. We selected the learning rate
from {1e-5, 1e-6, 3e-5, 3e-6, 5e-5, 5e-6} on **Marathi dev** F1 and then used
the same value (3e-5) for all languages. We also chose the early-stopping
rule (relative dev-loss improvement < 0.5% for 3 consecutive epochs) on
Marathi dev.

Projection weights (α=0.6, β=0.2, γ=0.2, λ=0.15, θ=0.8, τ=0.40, K=2):
TODO — say how these were chosen (grid search on which data? or set a priori?).

## Evaluation protocol

* Metric: span-level exact-match micro P/R/F1 over PER/LOC/ORG
  (`src/naamasetu/labels.py`, unit-tested in `tests/test_labels.py`).
* The test set is the Naamapadam gold test split. We never use it for LR or
  checkpoint selection.
* We report mean ± std over 5 training seeds: 42, 123, 456, 789, 2024.

## Compute

| Stage | Hardware | Wall time per language | Total GPU hours |
|---|---|---|---|
| 01 parallel extraction | TODO | TODO | TODO |
| 02 projection (~250k pairs) | TODO GPU | TODO | TODO |
| 05 fine-tuning (1 seed, 1 LR) | TODO GPU | TODO | TODO |

Software: Python 3.10, PyTorch TODO, Transformers TODO, CUDA TODO.
(Paste `pip freeze` into `requirements-lock.txt` for the camera-ready.)

## Known sources of variance

* spaCy English NER errors propagate into the silver labels.
* Wikipedia content changes over time. We release our crawled snapshot
  (crawl dates: TODO) so that Stage 1 does not have to be re-run.
* GPU non-determinism in fp16 training. Seed std is reported in the table.
