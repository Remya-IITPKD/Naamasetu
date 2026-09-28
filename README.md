# Naamasetu: Named Entity Recognition in Low-Resource Indic Languages

Code and data for the ARR submission *"Naamasetu: Named Entity Recognition in Low-Resource Indic Languages"* (anonymous).

**What this repo does.** We mine English–Indic parallel sentences from comparable Wikipedia articles and tag the English side with an off-the-shelf NER model. We then project those labels onto nine Indic languages with a **hybrid word aligner**, which combines contextual semantic similarity, phonetic (Metaphone) similarity and romanised-spelling similarity, plus an entity bonus. Finally, we fine-tune mBERT and XLM-R on the *gold* Naamapadam training set augmented with this *silver* data for PER/LOC/ORG NER, and compare with the published Naamapadam baseline.

Languages: Assamese (as), Gujarati (gu), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te).

## At a glance

| You want… | Look at |
|---|---|
| The paper's results tables | [`results/paper_tables.md`](results/paper_tables.md) (also below, under Results) |
| Every score for every model | [`results/test_scores.csv`](results/test_scores.csv) |
| Hyperparameters of each reported model | [`results/selected_runs.tsv`](results/selected_runs.tsv), [`configs/hyperparameters.yaml`](configs/hyperparameters.yaml) |
| The silver data and merged training sets | `data/projected/`, `data/augmented/` (see [`data/README.md`](data/README.md)) |
| Compute, software versions, evaluation details | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| To retrain a model on the released data | [Quick start](#quick-start) below |

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
        │  06_evaluate.py             type-, span- and token-level P/R/F1 on the gold test set
        │  07_aggregate_results.py    → results/main_table*.{md,tex}
        ▼
 results/
```

## Repository layout

```
.
├── src/naamasetu/            shared library
│   ├── languages.py         language registry (scripts, codes, Unicode blocks)
│   ├── labels.py            BIO schema + type-, span- and token-level P/R/F1
│   ├── significance.py      paired bootstrap test (optional tool)
│   └── data.py              JSONL I/O, sub-word label alignment
├── scripts/                 one entry point per stage (run in order)
│   ├── 00_prepare_naamapadam.py
│   ├── 01_extract_parallel.py
│   ├── 02_project_labels.py
│   ├── 03_filter_silver.py
│   ├── 03b_consistency_filter.py  (optional) keep silver sentences a gold-trained model agrees with
│   ├── 04_merge_silver_gold.py
│   ├── 05_train_ner.py
│   ├── 06_evaluate.py
│   ├── 07_aggregate_results.py
│   ├── 08_bootstrap_significance.py  (optional) paired test between two models
│   └── analysis/            alignment coverage + intrinsic comparisons
├── configs/hyperparameters.yaml   every value used for reported numbers
├── slurm/                   batch templates (projection, training)
├── run_pipeline.sh          end-to-end for one language
├── data/                    released silver + merged data (see data/README.md)
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

Retrain and evaluate an augmented model from the released data (Marathi, XLM-R):

```bash
# gold Naamapadam splits -> data/naamapadam/mr/{train,dev,test}.jsonl
python scripts/00_prepare_naamapadam.py --langs mr

# released merged training set (gold train + silver)
gunzip -k data/augmented/mr/train_aug_mr.jsonl.gz

python scripts/05_train_ner.py --lang mr --model-path xlm-roberta-base \
    --train data/augmented/mr/train_aug_mr.jsonl \
    --dev   data/naamapadam/mr/dev.jsonl \
    --test  data/naamapadam/mr/test.jsonl \
    --seed 42 --output-dir runs/xlmr/augmented/mr/seed42
```

The same on a SLURM cluster (use `MODEL=mbert` for mBERT):

```bash
sbatch --array=0 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=augmented slurm/train.slurm
```

The reported models were selected on the gold dev split plus 0.5% held-out
silver data; that merged dev file is not released, so the commands above use
the gold dev split, and the selected learning rate can occasionally differ.

The full pipeline from Wikipedia (`run_pipeline.sh`, `slurm/project.slurm`)
also needs the parallel sentence pairs (Stages 1–2), which are not included in
this release.

## Reproducing the main table

| Step | Command | Output |
|---|---|---|
| Augmented (gold + silver) | `slurm/train.slurm` with `SETTING=augmented`, `MODEL=mbert` / `xlmr` | `runs/*/augmented/*/seed42/` |
| Test evaluation | `scripts/06_evaluate.py --save-predictions` on each dev-selected `final_model` | metrics + predictions |
| Table | `python scripts/07_aggregate_results.py --metric test_type_f1` | `results/main_table_type.{md,tex}` |

All numbers are **F1 on the Naamapadam gold test sets**; the paper's tables use the type-level metric (see `docs/REPRODUCIBILITY.md`). The learning rate is selected **on dev only**; the test set is used once. Each configuration was trained with one seed (42).

## Results

NER F1 × 100 on the Naamapadam gold test sets. Augmented = Naamapadam training data +
Naamasetu silver data; one model per cell (seed 42, learning rate chosen on dev, see
`results/selected_runs.tsv`). All metrics and per-type scores: `results/test_scores.csv`.

**Main results** (paper layout; word-level micro-F1, B-/I- merged, `O` excluded)

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

Other metrics computed from the same predictions (`src/naamasetu/labels.py`):

* **span** (CoNLL exact match: boundaries and type must be right): `results/main_table.md`

| Model | Setting | as | gu | kn | ml | mr | or | pa | ta | te | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mbert | augmented | 48.00 | 77.09 | 78.17 | 77.83 | 79.87 | 27.45 | 69.60 | 70.14 | 79.17 | 67.48 |
| xlmr | augmented | 50.00 | 78.45 | 78.46 | 77.97 | 79.94 | 41.17 | 68.90 | 70.17 | 78.76 | 69.31 |

* **token** (micro F1 over the non-O BIO tags): `results/main_table_token.md`

| Model | Setting | as | gu | kn | ml | mr | or | pa | ta | te | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mbert | augmented | 67.47 | 81.16 | 83.34 | 81.71 | 84.58 | 27.64 | 77.08 | 75.42 | 84.56 | 73.66 |
| xlmr | augmented | 72.09 | 82.23 | 83.57 | 82.44 | 84.28 | 49.80 | 77.54 | 75.99 | 84.82 | 76.97 |

The Assamese test set has 51 sentences (24 entities) and the Odia one is also small, so
their scores are uncertain.

## Data

The Naamasetu silver data and the merged training sets for all nine languages are included in this repository, gzipped, under `data/projected/` and `data/augmented/`. See [`data/README.md`](data/README.md) for the layout, formats and statistics.

## Licence

Code: MIT (see `LICENSE`). Released data: see `data/README.md`. Our data is derived from Wikipedia, so it is CC BY-SA 4.0.

## Citation

Withheld for anonymous review.
