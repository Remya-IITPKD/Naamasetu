# Results in the paper's table layout

Metric: word-level micro-F1 over PER/ORG/LOC with B-/I- merged, `O` excluded
(`type_*` in `test_scores.csv`; the `entity_micro_F1` printed by the training scripts).
Augmented = Naamapadam + Naamasetu silver data. Every model: seed 42, learning rate chosen
on dev (`selected_runs.tsv`), tested once. Higher augmented F1 per language in bold.

## Table 6: NER F1 (%)

| Language | Naamapadam mBERT (cited) | Augmented mBERT | Augmented XLM-R |
|---|---|---|---|
| Malayalam | 81.49 | 83.71 | **84.94** |
| Assamese | 45.37 | 67.47 | **72.09** |
| Marathi | 81.37 | **86.61** | 86.26 |
| Odia | 25.01 | 30.49 | **55.00** |
| Gujarati | 80.59 | 83.24 | **84.08** |
| Kannada | 80.33 | 85.26 | **85.53** |
| Punjabi | 71.51 | 81.42 | **81.50** |
| Telugu | 82.49 | **87.04** | 86.98 |
| Tamil | 73.36 | 78.12 | **78.91** |

Naamapadam mBERT: reported by Mhaske et al. (2023), Table 5 (monolingual mBERT), not re-run. Because this baseline is cited rather than retrained, the difference to our models reflects the whole system (training data, encoder and training setup), not the silver data alone.

Differences of at most 0.07 from the training logs come from fp32 evaluation here vs. fp16 in the logs.

## Table 7: per-entity F1 (%) of the augmented models

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
