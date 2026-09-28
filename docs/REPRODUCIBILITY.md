# Reproducibility notes

This file collects what the ARR Responsible NLP Checklist asks about
(sections B–C) so reviewers can find it in one place.

## Models

| Model | HF id | Parameters |
|---|---|---|
| mBERT | `bert-base-multilingual-cased` | ~178M |
| XLM-R base | `xlm-roberta-base` | ~278M |
| Sentence encoder (Stage 1) | `paraphrase-multilingual-mpnet-base-v2` | ~278M |
| English NER (Stage 2) | spaCy 3.x `en_core_web_sm` (exact version not recorded) | – |

## Parallel-sentence threshold

Stage 1 keeps English–Indic sentence pairs with cosine similarity ≥ 0.70
(`configs/hyperparameters.yaml`). This is slightly lower than the 0.75 LaBSE
threshold used to mine Samanantar (Ramesh et al., 2022), because our pairs come
from comparable, independently written Wikipedia articles: they share entities
but rarely share exact wording, and a stricter threshold would discard many
useful pairs.

## Hyperparameters

All values are in `configs/hyperparameters.yaml`.

* **Augmented models (Naamapadam + silver), mBERT and XLM-R:** batch size 16,
  learning rate swept over {1e-5, 1e-6, 3e-5, 3e-6, 5e-5, 5e-6}. The LR is
  chosen **separately for each language and model on dev F1** (dev set =
  Naamapadam gold dev + 0.5% held-out silver). Early stopping: relative dev-loss
  improvement < 0.5% for 3 consecutive epochs, max 10 epochs.
* Weight decay 0.01, warmup ratio 0.1, fp16, max sequence length 256, seed 42.
  Early stopping only decides when training ends; the checkpoint with the
  highest dev F1 is restored and evaluated. (The `best_epoch` in the training
  logs is the lowest-dev-loss epoch, which can differ.)
* One training seed (42) per configuration; the LR sweeps already take about
  230 GPU-hours.
* The LR selected for every run is listed in `results/selected_runs.tsv`.

Projection weights (α=0.6, β=0.2, γ=0.2, λ=0.15, θ=0.8, τ=0.40, K=2) were
chosen empirically: several settings were tried on sample sentence pairs and
the one that aligned entity mentions most accurately was kept. The same
values are used for every language.

## Evaluation protocol

* Three metrics are computed from the same predictions
  (`src/naamasetu/labels.py`, unit-tested in `tests/test_labels.py`):
  * **type-level** micro P/R/F1: word by word after merging B-/I- into
    PER/LOC/ORG, `O` excluded (`labels.type_token_prf`). This is the metric of
    the paper's results tables and the `entity_micro_F1` the training scripts
    print. It is word-level, not entity-level, and the most lenient of the three;
  * **span-level** exact-match micro P/R/F1 (`labels.span_prf`);
  * **token-level** micro P/R/F1 over the non-O BIO tags (`labels.token_prf`).
* Dev selection used token-level F1 on the dev set (the Trainer's `f1`).
* Test numbers were produced with `scripts/06_evaluate.py` on the saved
  dev-selected models, so every reported number uses the same metric code.
* `scripts/08_bootstrap_significance.py` implements a paired bootstrap test
  between two models' predictions on the same test set. It is provided as a
  tool; the released tables compare against a cited baseline, for which no
  per-sentence predictions exist, so no significance test is reported.

### The cited Naamapadam baseline

The "Naamapadam mBERT" column is Table 5 of Mhaske et al. (2023) ("Mined data,
Awesome Align"): **monolingual** mBERT (uncased), one model per language,
reported on the same Naamapadam test sets. It is cited, not re-run. Our
augmented models differ from it in the training data (Naamapadam + silver), the
encoder (cased mBERT, XLM-R) and the training setup (LR sweep, relative-loss
early stopping). The difference between the columns therefore measures the
whole system, not the contribution of the silver data alone.

## Compute

| Stage | Hardware | Wall time per language | Total GPU hours |
|---|---|---|---|
| 01 parallel extraction | Google Colab | not recorded | not recorded |
| 02 projection | 1× NVIDIA H100 or 4–8 CPU cores | minutes (as) to ~8 h (gu), from job logs | not recorded |
| 05 fine-tuning, one run (1 LR) | 1× NVIDIA H100 80GB | 2 min (as) – 4.4 h (ml) | – |
| 05 fine-tuning, all reported runs | 1× H100 per run | – | ≈ 233 |

Fine-tuning time per run, averaged over the LR sweep (seconds):

| Model | as | gu | kn | ml | mr | or | pa | ta | te |
|---|---|---|---|---|---|---|---|---|---|
| XLM-R augmented | 194 | 8391 | 8942 | 15700 | 8036 | 3304 | 9148 | 11072 | 12032 |
| mBERT augmented | 144 | 8315 | 7236 | 12433 | 6461 | 2825 | 7672 | 8277 | 9921 |

GPU-hour total: 6 runs × 9 languages for each model (≈ 128 h XLM-R, ≈ 105 h
mBERT). Evaluating all dev-selected models on test takes under an hour on CPU.

Software (training and evaluation): Python 3.10.19, PyTorch 2.10.0 (CUDA 12.8),
Transformers 4.35.2, Datasets 2.14.6, NumPy 1.26.4.

## Known sources of variance

* spaCy English NER errors propagate into the silver labels.
* Wikipedia content changes over time. We release our crawled snapshot
  (Wikipedia `latest` dumps downloaded between March and June 2026) so that
  Stage 1 does not have to be re-run.
* GPU non-determinism in fp16 training; each configuration was trained once.
