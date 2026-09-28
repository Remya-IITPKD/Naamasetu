# Results

All files here are generated from the dev-selected models (see
`docs/REPRODUCIBILITY.md`); nothing is hand-edited.

| File | Contents | Produced by |
|---|---|---|
| `selected_runs.tsv` | batch size, LR and dev F1 of the model chosen for each model / setting / language | training logs (selection on dev) |
| `main_table.{md,tex}`, `summary.csv` | span-level test F1 × 100, with the augmented − gold Δ | `07_aggregate_results.py` |
| `main_table_token.{md,tex}`, `summary_token.csv` | the same for token-level F1 | `07_aggregate_results.py --metric test_token_f1` |
| `test_span_scores.csv` | span P/R/F1, token P/R/F1 and per-type (PER/LOC/ORG) P/R/F1/support | `06_evaluate.py` |
| `significance_xlmr.{md,csv}` | paired bootstrap, span F1: Δ, 95% CI, one-sided p | `08_bootstrap_significance.py` |
| `significance_xlmr_token.{md,csv}` | the same for token F1 | `08_bootstrap_significance.py --metric token` |

Each configuration was trained once (seed 42), so `summary*.csv` has
`n_seeds = 1` and `std = 0`.

To regenerate: evaluate each model listed in `selected_runs.tsv` with
`scripts/06_evaluate.py --save-predictions`, write one
`runs/<model>/<setting>/<lang>/seed42/results.json` per model (keys
`test_span_f1`, `test_token_f1`), then run `07_aggregate_results.py` and
`08_bootstrap_significance.py`.
