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

All values are in `configs/hyperparameters.yaml`.

* **Augmented (gold + silver), mBERT and XLM-R:** batch size 16, learning rate
  swept over {1e-5, 1e-6, 3e-5, 3e-6, 5e-5, 5e-6}. The LR is chosen
  **separately for each language and model on dev F1** (the augmented dev set =
  Naamapadam gold dev + 0.5% held-out silver). Early stopping: relative dev-loss
  improvement < 0.5% for 3 consecutive epochs, max 10 epochs.
* **Gold-only baseline, XLM-R:** grid of batch size {8, 16, 32} × the same 6
  LRs, chosen on the Naamapadam gold dev F1. Early stopping: dev F1 did not
  improve for 3 epochs, max 10 epochs.
* Both: weight decay 0.01, warmup ratio 0.1, fp16, max sequence length 256,
  seed 42, best epoch restored by dev F1.
* The LR / batch size selected for every run is listed in
  `results/selected_runs.tsv`.
* mBERT was trained on the augmented data only; there is no mBERT gold-only
  baseline, so the gold-vs-augmented comparison is reported for XLM-R.

Projection weights (α=0.6, β=0.2, γ=0.2, λ=0.15, θ=0.8, τ=0.40, K=2):
TODO — say how these were chosen (grid search on which data? or set a priori?).

## Evaluation protocol

* Primary metric: span-level exact-match micro P/R/F1 over PER/LOC/ORG
  (`src/naamasetu/labels.span_prf`, unit-tested in `tests/test_labels.py`).
* Also reported: token-level micro P/R/F1 over the non-O tags
  (`labels.token_prf`), i.e. sklearn micro F1 on first-sub-token labels. This is
  the metric the training loop logs as `test_f1` and uses for dev selection; it
  is higher than span F1 because partly correct entities earn partial credit.
  Always state which one a number is.
* The test set is the Naamapadam gold test split. We never use it for LR or
  checkpoint selection.
* The two XLM-R settings are evaluated on byte-identical test files.
* Each configuration was trained once (seed 42). Instead of seed averaging,
  we test whether the augmented model beats the gold-only model with a
  **paired bootstrap** over test sentences (10,000 resamples;
  `scripts/08_bootstrap_significance.py`), reporting the F1 gain, its 95% CI
  and a one-sided p-value per language (`results/significance_xlmr.md`).
  This measures test-set sampling variance, not training-seed variance.
* Test numbers were produced with `scripts/06_evaluate.py` on the saved
  dev-selected models, so every reported number uses the same metric code.

## Compute

| Stage | Hardware | Wall time per language | Total GPU hours |
|---|---|---|---|
| 01 parallel extraction | TODO (Colab) | TODO | TODO |
| 02 projection | 1× NVIDIA H100 or 4–8 CPU cores | minutes (as) to ~8 h (gu), from job logs | TODO |
| 05 fine-tuning, one run (1 LR) | 1× NVIDIA H100 80GB | 3 min (as) – 4.4 h (ml) | – |
| 05 fine-tuning, all reported runs | 1× H100 per run | – | ≈ 575 |

Fine-tuning time per run, averaged over the LR sweep (seconds):

| Model / setting | as | gu | kn | ml | mr | or | pa | ta | te |
|---|---|---|---|---|---|---|---|---|---|
| XLM-R augmented | 194 | 8391 | 8942 | 15700 | 8036 | 3304 | 9148 | 11072 | 12032 |
| mBERT augmented | 144 | 8315 | 7236 | 12433 | 6461 | 2825 | 7672 | 8277 | 9921 |
| XLM-R gold only | 167 | 8291 | 7982 | 15817 | 6871 | 3321 | 7605 | 8666 | 9750 |

GPU-hour total: 6 runs × 9 languages for each augmented model (≈ 128 h XLM-R,
≈ 105 h mBERT) plus the 18-run gold grid (≈ 340 h). Evaluating all
dev-selected models on test takes under an hour on CPU.

Software (augmented runs and all evaluation): Python 3.10.19, PyTorch 2.10.0
(CUDA 12.8), Transformers 4.35.2, Datasets 2.14.6, NumPy 1.26.4. The gold-only
runs saved their tokenizer with a newer `tokenizers` release; its vocabulary is
identical to `xlm-roberta-base`, so `06_evaluate.py --tokenizer xlm-roberta-base`
loads those models under the versions above.

## Known sources of variance

* spaCy English NER errors propagate into the silver labels.
* Wikipedia content changes over time. We release our crawled snapshot
  (crawl dates: TODO) so that Stage 1 does not have to be re-run.
* GPU non-determinism in fp16 training. Seed std is reported in the table.
