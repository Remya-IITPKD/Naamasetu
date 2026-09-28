# Results

All files here are generated from the dev-selected models (see
`docs/REPRODUCIBILITY.md`); nothing is hand-edited.

| File | Contents | Produced by |
|---|---|---|
| `selected_runs.tsv` | batch size, LR and dev F1 of the model chosen for each model / setting / language | training logs (selection on dev) |
| `main_table.{md,tex}`, `summary.csv` | span-level test F1 × 100, with the augmented − gold Δ | `07_aggregate_results.py` |
| `main_table_token.{md,tex}`, `summary_token.csv` | the same for token-level (BIO) F1 | `07_aggregate_results.py --metric test_token_f1` |
| `main_table_type.{md,tex}`, `summary_type.csv` | the same for type-level F1 (B/I merged) | `07_aggregate_results.py --metric test_type_f1` |
| `test_scores.csv` | every metric (span, token, type) with P/R/F1, plus per-type P/R/F1/support for span and type | `06_evaluate.py` |
| `significance_xlmr{,_token,_type}.{md,csv}` | paired bootstrap: Δ, 95% CI, one-sided p, per metric | `08_bootstrap_significance.py --metric span/token/type` |

Each configuration was trained once (seed 42), so `summary*.csv` has
`n_seeds = 1` and `std = 0`.

To regenerate: evaluate each model listed in `selected_runs.tsv` with
`scripts/06_evaluate.py --save-predictions`, write one
`runs/<model>/<setting>/<lang>/seed42/results.json` per model (keys
`test_span_f1`, `test_token_f1`, `test_type_f1`), then run `07_aggregate_results.py` and
`08_bootstrap_significance.py`.
