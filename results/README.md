# Results

The NER files here are generated from the dev-selected models (see
`docs/REPRODUCIBILITY.md`) and the alignment files from the released alignments;
nothing is hand-edited.

| File | Contents | Produced by |
|---|---|---|
| `paper_tables.md` | the paper's main-results and per-entity tables, in the paper's layout and metric | built from `test_scores.csv` |
| `selected_runs.tsv` | batch size, LR and dev F1 of the model chosen for each model and language | training logs (selection on dev) |
| `main_table.{md,tex}`, `summary.csv` | span-level test F1 × 100 | `07_aggregate_results.py` |
| `main_table_token.{md,tex}`, `summary_token.csv` | the same for token-level (BIO) F1 | `07_aggregate_results.py --metric test_token_f1` |
| `main_table_type.{md,tex}`, `summary_type.csv` | the same for type-level F1 (B/I merged) | `07_aggregate_results.py --metric test_type_f1` |
| `test_scores.csv` | every metric (span, token, type) with P/R/F1, plus per-type P/R/F1/support for span and type | `06_evaluate.py` |
| `alignment_eval/` | alignment evaluation: source/target coverage and entity-word coverage of SimAlign, awesome-align and the hybrid aligner on 200 pairs per language (paper Tables 2 and 6); see its README | `scripts/analysis/build_alignment_sample.py`, `alignment_eval_scores.py`, `alignment_tables.py` |

Each configuration was trained once (seed 42), so `summary*.csv` has
`n_seeds = 1` and `std = 0`.

To regenerate: evaluate each model listed in `selected_runs.tsv` with
`scripts/06_evaluate.py --save-predictions`, write one
`runs/<model>/<setting>/<lang>/seed42/results.json` per model (keys
`test_span_f1`, `test_token_f1`, `test_type_f1`), then run `07_aggregate_results.py`.
