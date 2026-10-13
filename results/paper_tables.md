# Results in the paper's table layout

Metric: word-level micro-F1 over PER/ORG/LOC with B-/I- merged, `O` excluded
(`type_*` in `test_scores.csv`; the `entity_micro_F1` printed by the training scripts).
Augmented = Naamapadam + Naamasetu silver data. Every model: seed 42, learning rate chosen
on dev (`selected_runs.tsv`), tested once. Higher augmented F1 per language in bold.

## Table 8: NER F1 (%), rounded to whole numbers

Same values as Table 16 below, rounded half-up. Higher augmented F1 per language in bold;
Telugu rounds to 87 for both encoders, and mBERT is bold because it is higher before
rounding (87.04 vs 86.98).

| Language | Baseline mBERT† | Naamasetu only mBERT | Augmented mBERT | Augmented XLM-R |
|---|---|---|---|---|
| Malayalam | 81 | 74 | 84 | **85** |
| Assamese | 45 | 22 | 67 | **72** |
| Marathi | 81 | 62 | **87** | 86 |
| Odia | 25 | 9 | 30 | **55** |
| Gujarati | 81 | 72 | 83 | **84** |
| Kannada | 80 | 73 | 85 | **86** |
| Punjabi | 72 | 68 | 81 | **82** |
| Telugu | 82 | 73 | **87** | 87 |
| Tamil | 73 | 64 | 78 | **79** |
| Average | 69.1 | 57.4 | 75.9 | 79.5 |

† Naamapadam mBERT, cited from Mhaske et al. (2023), not re-run.

Statements in Section 5.3 that follow from these numbers:

* Naamasetu-only mBERT reaches 83% of the Naamapadam baseline on average
  (57.43 / 69.06 = 0.832).
* Naamasetu is about 20% of Naamapadam's size: 767,727 entity-bearing sentences vs.
  3,820,084 Naamapadam training sentences over the nine languages (Table 3),
  767,727 / 3,820,084 = 0.201.

## Main results: NER F1 (%) (paper Table 16; Table 8 shows the same values rounded)

| Language | Naamapadam mBERT (cited) | Naamasetu only mBERT | Naamasetu only XLM-R | Augmented mBERT | Augmented XLM-R | Relative F1 mBERT | Relative F1 XLM-R |
|---|---|---|---|---|---|---|---|
| Malayalam | 81.49 | 73.54 | 77.66 | 83.71 | 84.94 | 88 | 91 |
| Assamese | 45.37 | 21.57 | 15.79 | 67.47 | 72.09 | 32 | 22 |
| Marathi | 81.37 | 62.28 | 67.32 | 86.61 | 86.26 | 72 | 78 |
| Odia | 25.01 | 9.12 | 16.18 | 30.49 | 55.00 | 30 | 29 |
| Gujarati | 80.59 | 71.96 | 80.74 | 83.24 | 84.08 | 86 | 96 |
| Kannada | 80.33 | 73.11 | 78.58 | 85.26 | 85.53 | 86 | 92 |
| Punjabi | 71.51 | 68.29 | 71.40 | 81.42 | 81.50 | 84 | 88 |
| Telugu | 82.49 | 72.79 | 78.28 | 87.04 | 86.98 | 84 | 90 |
| Tamil | 73.36 | 64.20 | 68.07 | 78.12 | 78.91 | 82 | 86 |
| Average | 69.06 | 57.43 | 61.56 | 75.93 | 79.48 | 76 | 77 |

Naamasetu only = trained on the Naamasetu silver data alone (`silver_only/`, built with
`scripts/09_silver_only_split.py`). Relative F1 = Naamasetu-only F1 / Augmented F1 × 100 for
the same encoder; the Average row uses the average F1 values.

Naamapadam mBERT: reported by Mhaske et al. (2023), Table 5 (monolingual mBERT), not re-run. Because this baseline is cited rather than retrained, the difference to our models reflects the whole system (training data, encoder and training setup), not the silver data alone.

Differences of at most 0.07 from the training logs come from fp32 evaluation here vs. fp16 in the logs.

## Per-entity F1 (%) of the augmented models

| Language | mBERT PER | mBERT LOC | mBERT ORG | XLM-R PER | XLM-R LOC | XLM-R ORG |
|---|---|---|---|---|---|---|
| Malayalam | 91.49 | 84.44 | 62.87 | 92.54 | 84.36 | 65.90 |
| Assamese | 82.35 | 44.44 | 70.83 | 90.00 | 50.00 | 73.91 |
| Marathi | 93.10 | 85.93 | 75.08 | 93.61 | 85.28 | 73.99 |
| Odia | 38.54 | 36.94 | 7.19 | 62.36 | 58.03 | 38.92 |
| Gujarati | 92.48 | 81.22 | 71.30 | 92.97 | 82.01 | 72.59 |
| Kannada | 91.72 | 82.16 | 75.67 | 92.97 | 80.65 | 74.86 |
| Punjabi | 89.72 | 78.80 | 72.78 | 89.72 | 79.80 | 72.50 |
| Telugu | 92.23 | 83.58 | 79.05 | 91.89 | 83.80 | 79.45 |
| Tamil | 85.63 | 78.07 | 69.01 | 87.12 | 78.71 | 69.06 |
