# Appendix results

Every table in the paper appendix, regenerated from the released data and logs. CSVs in this folder are written by the scripts named below.

## Paper Table 1: dataset comparison

See [`dataset_comparison.md`](dataset_comparison.md) (sentences per language in existing Indic NER datasets, with the source of every number).

## A. Corpus statistics

### Parallel page and sentence pairs (`page_stats.csv`, `scripts/analysis/page_stats.py`)

Page pairs = Indic–English Wikipedia page pairs that yielded at least one parallel sentence (context similarity ≥ 0.7).

| Language | Page pairs | Sentence pairs | Min / page | Max / page | Avg / page |
|---|---|---|---|---|---|
| Malayalam | 49,686 | 240,153 | 1 | 210 | 4.8 |
| Tamil | 83,712 | 299,637 | 1 | 218 | 3.6 |
| Telugu | 50,143 | 360,231 | 1 | 285 | 7.2 |
| Kannada | 13,976 | 105,592 | 1 | 345 | 7.6 |
| Marathi | 22,170 | 82,776 | 1 | 353 | 3.7 |
| Gujarati | 7,973 | 75,005 | 1 | 301 | 9.4 |
| Punjabi | 29,754 | 128,364 | 1 | 205 | 4.3 |
| Odia | 7,680 | 23,010 | 1 | 112 | 3.0 |
| Assamese | 1,004 | 1,350 | 1 | 13 | 1.3 |
| Total | 266,098 | 1,316,118 | 1 | 353 | 4.9 |

### Entity strings new to Naamapadam (`new_entities.csv`, `scripts/analysis/new_entities.py`)

Distinct (type, string) entity spans in the silver data whose string never occurs as an entity in Naamapadam train + dev + test (exact match after NFC normalisation; inflected forms count as new). Needs `data/naamapadam/` from `scripts/00_prepare_naamapadam.py`.

| Language | Distinct | Not in Naamapadam | % new | % new PER | % new LOC | % new ORG |
|---|---|---|---|---|---|---|
| Malayalam | 170,385 | 135,228 | 79.4 | 79.3 | 73.2 | 82.8 |
| Tamil | 216,264 | 169,646 | 78.4 | 79.9 | 70.5 | 81.3 |
| Telugu | 243,812 | 200,043 | 82.0 | 82.9 | 74.6 | 84.6 |
| Kannada | 54,225 | 39,084 | 72.1 | 70.1 | 64.4 | 76.7 |
| Marathi | 44,689 | 30,307 | 67.8 | 63.5 | 61.8 | 73.9 |
| Gujarati | 65,583 | 43,710 | 66.6 | 67.4 | 58.2 | 70.9 |
| Punjabi | 89,904 | 63,423 | 70.5 | 71.2 | 58.4 | 74.5 |
| Odia | 19,559 | 14,172 | 72.5 | 74.3 | 64.6 | 75.0 |
| Assamese | 1,325 | 1,227 | 92.6 | 94.7 | 86.4 | 92.7 |
| Total | 905,746 | 696,840 | 76.9 | | | |

## B. Alignment analysis (`classwise_alignment.csv`, `scripts/analysis/classwise_alignment.py`)

Same 200 sentence pairs per language as `results/alignment_eval`. Coverage = fraction of English entity words of the class with at least one link. Similarity = embedding cosine for SimAlign and awesome-align, the combined hybrid score H for the hybrid aligner (different measures; compare within a column).

| Lang | Coverage SimAlign P/L/O | Coverage awesome P/L/O | Coverage hybrid P/L/O | Sim SimAlign | Sim awesome | Sim hybrid (H) |
|---|---|---|---|---|---|---|
| ml | 0.848/0.820/0.862 | 0.500/0.414/0.455 | 0.854/0.684/0.728 | 0.53/0.50/0.55 | 0.67/0.66/0.68 | 0.81/0.81/0.82 |
| mr | 0.814/0.870/0.748 | 0.581/0.652/0.478 | 0.872/0.942/0.807 | 0.68/0.66/0.66 | 0.72/0.69/0.70 | 0.84/0.83/0.82 |
| kn | 0.726/0.852/0.796 | 0.548/0.444/0.552 | 0.887/0.880/0.912 | 0.67/0.65/0.67 | 0.71/0.70/0.72 | 0.83/0.83/0.83 |
| gu | 0.760/0.782/0.670 | 0.408/0.506/0.306 | 0.872/0.955/0.874 | 0.62/0.62/0.60 | 0.66/0.66/0.66 | 0.80/0.80/0.79 |
| pa | 0.963/0.964/0.868 | 0.770/0.703/0.560 | 1.000/1.000/0.985 | 0.68/0.67/0.64 | 0.71/0.71/0.68 | 0.84/0.84/0.82 |
| ta | 0.832/0.797/0.778 | 0.545/0.464/0.425 | 0.993/0.891/0.923 | 0.67/0.67/0.66 | 0.71/0.70/0.71 | 0.83/0.83/0.82 |
| te | 0.904/0.935/0.902 | 0.730/0.591/0.611 | 0.991/0.957/0.930 | 0.70/0.65/0.66 | 0.73/0.69/0.70 | 0.84/0.83/0.83 |
| or | 0.587/0.420/0.404 | 0.321/0.227/0.235 | 0.891/0.818/0.826 | 0.76/0.59/0.66 | 0.78/0.69/0.73 | 0.82/0.75/0.76 |
| as | 0.853/0.745/0.775 | 0.578/0.509/0.537 | 0.905/0.927/0.840 | 0.67/0.67/0.70 | 0.75/0.72/0.77 | 0.83/0.82/0.85 |

## C. Downstream NER

### Training dynamics (`training_dynamics.csv`, `scripts/analysis/training_dynamics.py`)

From the per-epoch logs in `results/training_logs/` (selected augmented runs, batch 16). Kept = epoch with the highest development F1 (the restored checkpoint).

| Model | Lang | LR | Epochs | Min dev loss (epoch) | Kept epoch | Dev F1 | Test F1 |
|---|---|---|---|---|---|---|---|
| xlmr | ml | 3e-05 | 8 | 0.243 (5) | 7 | 79.14 | 84.94 |
| xlmr | as | 1e-05 | 8 | 0.429 (5) | 7 | 47.34 | 72.09 |
| xlmr | mr | 5e-05 | 7 | 0.224 (4) | 6 | 80.71 | 86.26 |
| xlmr | or | 5e-05 | 7 | 0.319 (5) | 7 | 43.56 | 55.00 |
| xlmr | gu | 5e-05 | 7 | 0.199 (4) | 7 | 80.49 | 84.08 |
| xlmr | kn | 5e-05 | 8 | 0.25 (5) | 8 | 79.51 | 85.53 |
| xlmr | pa | 3e-05 | 7 | 0.207 (4) | 6 | 75.56 | 81.50 |
| xlmr | te | 3e-05 | 8 | 0.252 (5) | 7 | 75.35 | 86.98 |
| xlmr | ta | 1e-05 | 8 | 0.272 (5) | 6 | 72.47 | 78.91 |
| mbert | ml | 3e-05 | 8 | 0.247 (5) | 8 | 78.91 | 83.71 |
| mbert | as | 5e-06 | 7 | 0.492 (4) | 4 | 43.68 | 67.47 |
| mbert | mr | 1e-05 | 7 | 0.225 (4) | 4 | 80.25 | 86.61 |
| mbert | or | 1e-05 | 6 | 0.419 (4) | 2 | 20.9 | 30.49 |
| mbert | gu | 5e-05 | 10 | 0.204 (7) | 9 | 80.22 | 83.24 |
| mbert | kn | 1e-05 | 7 | 0.249 (5) | 7 | 79.0 | 85.26 |
| mbert | pa | 5e-06 | 7 | 0.214 (4) | 7 | 75.37 | 81.42 |
| mbert | te | 3e-05 | 6 | 0.254 (3) | 6 | 74.72 | 87.04 |
| mbert | ta | 5e-06 | 6 | 0.282 (5) | 5 | 71.7 | 78.12 |

### Training on Naamasetu silver data only (`results/silver_only/`)

Same backbones trained on the silver data alone (the 99.5% silver split of the augmented runs, recreated by `scripts/09_silver_only_split.py`), no Naamapadam training data; LR chosen from {1e-5, 3e-5, 5e-5} on the Naamapadam dev set (`results/silver_only/selected_runs.tsv`); evaluated on the Naamapadam test sets with `scripts/06_evaluate.py` (type-level F1, as in the main results). Relative = silver-only F1 / augmented F1.

| Language | Naamapadam mBERT† | Silver-only mBERT | Silver-only XLM-R | Augmented mBERT | Augmented XLM-R | Relative mBERT | Relative XLM-R |
|---|---|---|---|---|---|---|---|
| Malayalam | 81.49 | 73.54 | 77.66 | 83.71 | 84.94 | 88% | 91% |
| Assamese | 45.37 | 21.57 | 15.79 | 67.47 | 72.09 | 32% | 22% |
| Marathi | 81.37 | 62.28 | 67.32 | 86.61 | 86.26 | 72% | 78% |
| Odia | 25.01 | 9.12 | 16.18 | 30.49 | 55.00 | 30% | 29% |
| Gujarati | 80.59 | 71.96 | 80.74 | 83.24 | 84.08 | 86% | 96% |
| Kannada | 80.33 | 73.11 | 78.58 | 85.26 | 85.53 | 86% | 92% |
| Punjabi | 71.51 | 68.29 | 71.40 | 81.42 | 81.50 | 84% | 88% |
| Telugu | 82.49 | 72.79 | 78.28 | 87.04 | 86.98 | 84% | 90% |
| Tamil | 73.36 | 64.20 | 68.07 | 78.12 | 78.91 | 82% | 86% |

† Cited from Mhaske et al. (2023), not re-run.
