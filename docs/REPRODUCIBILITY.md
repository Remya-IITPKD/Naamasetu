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

## Parallel-sentence threshold

Stage 1 keeps English–Indic sentence pairs with cosine similarity ≥ 0.70
(`configs/hyperparameters.yaml`). This is slightly lower than the 0.75 used
for translation-based parallel data, because our pairs come from comparable,
independently written Wikipedia articles: they share entities but rarely share
exact wording, and a stricter threshold would discard many useful pairs.

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
  seed 42. Early stopping only decides when training ends; the checkpoint
  with the highest dev F1 is restored and evaluated. (The `best_epoch` in the
  training logs is the lowest-dev-loss epoch, which can differ.)
* One seed only: the LR sweeps already take ≈ 575 GPU-hours, so the
  augmented-vs-gold difference is tested with a paired bootstrap instead.
* The LR / batch size selected for every run is listed in
  `results/selected_runs.tsv`.
* mBERT was trained on the augmented data only. The mBERT gold-only numbers
  are **taken from the Naamapadam paper** (Mhaskar et al., 2023), not
  reproduced. The XLM-R gold-vs-augmented comparison is fully controlled (same
  code, data version, selection and metric) and is the one tested for
  significance.

Projection weights (α=0.6, β=0.2, γ=0.2, λ=0.15, θ=0.8, τ=0.40, K=2):
TODO — say how these were chosen (grid search on which data? or set a priori?).

## Evaluation protocol

* Primary metric: span-level exact-match micro P/R/F1 over PER/LOC/ORG
  (`src/naamasetu/labels.span_prf`, unit-tested in `tests/test_labels.py`).
* Also reported, from the same predictions:
  * token-level micro P/R/F1 over the non-O BIO tags (`labels.token_prf`);
  * type-level micro P/R/F1 after merging B-/I- into PER/LOC/ORG
    (`labels.type_token_prf`). This is the `entity_micro_F1` the training
    scripts print. It is word-level, not entity-level, and is the most
    lenient of the three.
* Dev selection used token-level F1 on the dev set (the Trainer's `f1`).
  State which metric any reported number uses.
* The two XLM-R settings are evaluated on byte-identical test files.
* Each configuration was trained once (seed 42). Instead of seed averaging,
  we test whether the augmented model beats the gold-only model with a
  **paired bootstrap** over test sentences (10,000 resamples;
  `scripts/08_bootstrap_significance.py`), reporting the F1 gain, its 95% CI
  and a one-sided p-value per language (`results/significance_xlmr*.md`, one file per metric).
  This measures test-set sampling variance, not training-seed variance.
* Test numbers were produced with `scripts/06_evaluate.py` on the saved
  dev-selected models, so every reported number uses the same metric code.

### Caveats for the cited mBERT baseline

The published mBERT numbers and our augmented mBERT runs differ in more than
the training data:

* training code and hyperparameters (their fine-tuning setup vs. our LR sweep
  and relative-loss early stopping);
* the cited numbers are Table 5 of Mhaske et al. (2023) ("Mined data,
  Awesome Align"): **monolingual** mBERT (uncased), one model per language,
  reported on the same Naamapadam test sets (not their multilingual results);
  per the authors of this work the F1 definition matches the one in
  `main_table_type.md`, but the Naamapadam paper does not state it explicitly;
* no per-sentence predictions, so no significance test is possible.

Differences between the two rows are therefore indicative only.

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
