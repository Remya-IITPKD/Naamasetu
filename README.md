# Naamasetu: Cross-lingual Named Entity Transfer by Bridging Languages

Code and data for the ARR submission *"Naamasetu: A Cross-lingual Named Entity Transfer by Bridging Languages"* (anonymous).

**What this repo does.** We mine English–Indic parallel sentences from comparable Wikipedia articles and tag the English side with an off-the-shelf NER model. We then project those labels onto nine Indic languages with a **hybrid word aligner**, which combines contextual semantic similarity, phonetic (Metaphone) similarity and romanised-spelling similarity, plus an entity bonus. Finally, we check whether adding this *silver* data to the *gold* Naamapadam training set improves PER/LOC/ORG NER with mBERT and XLM-R.

Languages: Assamese (as), Gujarati (gu), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te).

---

## Pipeline

```
 Wikipedia title pairs
        │  01_extract_parallel.py     MPNet sentence embeddings + greedy 1-1 matching
        ▼
 EN–Indic parallel sentences
        │  02_project_labels.py       spaCy EN NER → hybrid alignment (Alg. 1) → BIO projection
        ▼
 silver Indic NER data
        │  03_filter_silver.py        drop all-O / malformed sentences
        │  04_merge_silver_gold.py    silver → train (+0.5% → dev); gold test untouched
        ▼
 gold + silver splits
        │  05_train_ner.py            mBERT / XLM-R, LR sweep chosen on dev
        ▼
 dev-selected model per language
        │  06_evaluate.py             span-level P/R/F1 on the gold test set
        │  07_aggregate_results.py    → results/main_table.{md,tex}
        │  08_bootstrap_significance.py  gold vs augmented: Δ, 95% CI, p
        ▼
 results/
```

## Repository layout

```
.
├── src/naamasetu/            shared library
│   ├── languages.py         language registry (scripts, codes, Unicode blocks)
│   ├── labels.py            BIO schema + span-level P/R/F1 (CoNLL exact match)
│   ├── significance.py      paired bootstrap test
│   └── data.py              JSONL I/O, sub-word label alignment
├── scripts/                 one entry point per stage (run in order)
│   ├── 00_prepare_naamapadam.py
│   ├── 01_extract_parallel.py
│   ├── 02_project_labels.py
│   ├── 03_filter_silver.py
│   ├── 04_merge_silver_gold.py
│   ├── 05_train_ner.py
│   ├── 06_evaluate.py
│   ├── 07_aggregate_results.py
│   ├── 08_bootstrap_significance.py
│   └── analysis/            alignment coverage + intrinsic comparisons
├── configs/hyperparameters.yaml   every value used for reported numbers
├── slurm/                   batch templates (projection, training)
├── run_pipeline.sh          end-to-end for one language
├── data/                    see data/README.md (formats + download links)
├── results/                 aggregated tables (generated)
├── docs/REPRODUCIBILITY.md  compute, runtimes, variance, artifact licences
└── tests/                   unit tests for metrics and projection
```

## Setup

```bash
conda env create -f environment.yml && conda activate naamasetu
pip install -e .
python -m spacy download en_core_web_sm
pytest -q tests          # sanity check (no GPU needed)
```

## Quick start

```bash
# 0) gold data
python scripts/00_prepare_naamapadam.py --langs mr

# 1–7) full pipeline for Marathi (uses the released parallel corpus)
bash run_pipeline.sh mr
```

On a SLURM cluster:

```bash
sbatch --export=ALL,TGT_LANG=mr slurm/project.slurm
sbatch --array=0 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=augmented slurm/train.slurm
sbatch --array=0 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=gold      slurm/train.slurm
python scripts/07_aggregate_results.py
```

## Reproducing the main table

| Step | Command | Output |
|---|---|---|
| Baseline (gold only) | `slurm/train.slurm` with `SETTING=gold` | `runs/*/gold/*/seed42/` |
| Augmented (gold + silver) | `slurm/train.slurm` with `SETTING=augmented` | `runs/*/augmented/*/seed42/` |
| Test evaluation | `scripts/06_evaluate.py --save-predictions` on each dev-selected `final_model` | metrics + predictions |
| Table | `python scripts/07_aggregate_results.py` | `results/main_table.{md,tex}` |
| Significance | `python scripts/08_bootstrap_significance.py --pair …` | `results/significance_xlmr.{md,csv}` |

All numbers are **span-level exact-match micro F1 on the Naamapadam gold test set**. Learning rate (and, for the gold baseline, batch size) is selected **on dev only**; the test set is used once. Each configuration was trained with one seed (42); significance comes from a paired bootstrap over test sentences (see `docs/REPRODUCIBILITY.md`).

## Results

Test F1 × 100 on the Naamapadam gold test set, one dev-selected model per cell
(seed 42). Selected hyperparameters: `results/selected_runs.tsv`. Per-type
scores: `results/test_span_scores.csv`.

**Span-level (exact match), primary metric**

| Model | Setting | as | gu | kn | ml | mr | or | pa | ta | te | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mbert | augmented | 48.00 | 77.09 | 78.17 | 77.83 | 79.87 | 27.45 | 69.60 | 70.14 | 79.17 | 67.48 |
| xlmr | gold | 40.00 | 77.77 | 78.54 | 77.40 | 78.38 | 39.00 | 68.46 | 66.75 | 80.90 | 67.47 |
| xlmr | augmented | 50.00 | 78.45 | 78.46 | 77.97 | 79.94 | 41.17 | 68.90 | 70.17 | 78.76 | 69.31 |
| xlmr | Δ | +10.00 | +0.68 | -0.08 | +0.57 | +1.56 | +2.17 | +0.43 | +3.42 | -2.14 | |

**Token-level (micro over non-O tags), for comparison with token-based reporting**

| Model | Setting | as | gu | kn | ml | mr | or | pa | ta | te | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mbert | augmented | 67.47 | 81.16 | 83.34 | 81.71 | 84.58 | 27.64 | 77.08 | 75.42 | 84.56 | 73.66 |
| xlmr | gold | 70.45 | 81.19 | 83.84 | 82.46 | 82.42 | 49.00 | 76.76 | 73.35 | 85.84 | 76.15 |
| xlmr | augmented | 72.09 | 82.23 | 83.57 | 82.44 | 84.28 | 49.80 | 77.54 | 75.99 | 84.82 | 76.97 |
| xlmr | Δ | +1.64 | +1.03 | -0.27 | -0.02 | +1.86 | +0.80 | +0.77 | +2.64 | -1.02 | |

**Is the XLM-R gain significant?** Paired bootstrap over test sentences
(10,000 resamples), one-sided p for "augmented > gold".

Span-level:

| Lang | Gold F1 | Aug F1 | Δ | 95% CI | p |
|---|---|---|---|---|---|
| as | 40.00 | 50.00 | +10.00 | [+0.85, +24.56] | 0.0189 |
| gu | 77.77 | 78.45 | +0.68 | [-0.61, +2.01] | 0.1529 |
| kn | 78.54 | 78.46 | -0.08 | [-1.63, +1.47] | 0.5394 |
| ml | 77.40 | 77.97 | +0.57 | [-1.07, +2.21] | 0.2467 |
| mr | 78.38 | 79.94 | +1.56 | [+0.27, +2.86] | 0.0082 |
| or | 39.00 | 41.17 | +2.17 | [-0.27, +4.60] | 0.0399 |
| pa | 68.46 | 68.90 | +0.43 | [-1.12, +1.96] | 0.2915 |
| ta | 66.75 | 70.17 | +3.42 | [+0.82, +6.07] | 0.0042 |
| te | 80.90 | 78.76 | -2.14 | [-4.15, -0.27] | 0.9873 |

Token-level:

| Lang | Gold F1 | Aug F1 | Δ | 95% CI | p |
|---|---|---|---|---|---|
| as | 70.45 | 72.09 | +1.64 | [-2.86, +8.84] | 0.2831 |
| gu | 81.19 | 82.23 | +1.03 | [-0.17, +2.26] | 0.0451 |
| kn | 83.84 | 83.57 | -0.27 | [-1.49, +0.93] | 0.6633 |
| ml | 82.46 | 82.44 | -0.02 | [-1.31, +1.26] | 0.5072 |
| mr | 82.42 | 84.28 | +1.86 | [+0.71, +3.05] | 0.0005 |
| or | 49.00 | 49.80 | +0.80 | [-1.99, +3.41] | 0.2698 |
| pa | 76.76 | 77.54 | +0.77 | [-0.53, +2.06] | 0.1202 |
| ta | 73.35 | 75.99 | +2.64 | [+0.68, +4.76] | 0.0046 |
| te | 85.84 | 84.82 | -1.02 | [-2.58, +0.42] | 0.9130 |

Notes:

* The Assamese (51 sentences) and Odia test sets are small, so their scores and
  CIs are very wide; treat those differences with caution.
* mBERT was trained on the augmented data only. The mBERT gold-only baseline
  in the paper is **cited from the Naamapadam paper** (Mhaskar et al., 2023),
  not re-run here, so it is not in these tables and has no significance test
  (that needs per-sentence predictions). See `docs/REPRODUCIBILITY.md` for the
  caveats of that comparison.
* Token-level F1 is always higher than span-level F1 because it gives credit
  for partly correct entities. Both come from the same predictions
  (`scripts/06_evaluate.py`).

## Data

See [`data/README.md`](data/README.md) for file formats, sizes and download links for the parallel corpora, the projected silver data and the merged splits.

## Licence

Code: MIT (see `LICENSE`). Released data: see `data/README.md`. Our data is derived from Wikipedia, so it is CC BY-SA 4.0.

## Citation

Withheld for anonymous review.
