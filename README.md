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
 gold + silver splits ───────────────┐
        │  05_train_ner.py            mBERT / XLM-R, relative-loss early stopping
        ▼                             │
 runs/<model>/<setting>/<lang>/seed*/results.json
        │  07_aggregate_results.py    mean ± std over 5 seeds → tables
        ▼
 results/main_table.{md,tex}
```

## Repository layout

```
.
├── src/naamasetu/            shared library
│   ├── languages.py         language registry (scripts, codes, Unicode blocks)
│   ├── labels.py            BIO schema + span-level P/R/F1 (CoNLL exact match)
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
│   └── analysis/            alignment coverage + intrinsic comparisons
├── configs/hyperparameters.yaml   every value used for reported numbers
├── slurm/                   batch templates (projection, training array over seeds)
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
sbatch --array=0-4 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=augmented slurm/train.slurm
sbatch --array=0-4 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=gold      slurm/train.slurm
python scripts/07_aggregate_results.py
```

## Reproducing the main table

| Step | Command | Output |
|---|---|---|
| Baseline (gold only) | `slurm/train.slurm` with `SETTING=gold` | `runs/*/gold/*/seed*/results.json` |
| Augmented (gold + silver) | `slurm/train.slurm` with `SETTING=augmented` | `runs/*/augmented/*/seed*/results.json` |
| Table | `python scripts/07_aggregate_results.py` | `results/main_table.{md,tex}` |

All numbers are **span-level exact-match micro F1 on the Naamapadam gold test set**, reported as mean ± std over 5 training seeds. Learning rate and checkpoints are selected **on dev only**.

## Results

<!-- TODO: paste results/main_table.md here once all runs finish -->
See `results/main_table.md`.

## Data

See [`data/README.md`](data/README.md) for file formats, sizes and download links for the parallel corpora, the projected silver data and the merged splits.

## Licence

Code: MIT (see `LICENSE`). Released data: see `data/README.md`. Our data is derived from Wikipedia, so it is CC BY-SA 4.0.

## Citation

Withheld for anonymous review.
