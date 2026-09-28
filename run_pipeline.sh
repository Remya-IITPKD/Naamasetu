#!/bin/bash
# End-to-end pipeline for ONE language on a single GPU machine.
#   bash run_pipeline.sh mr
# Stage 1 (Wikipedia crawling) is skipped by default because we release its
# output; set CRAWL=1 to re-run it from the title list.
set -euo pipefail
TGT_LANG=${1:?usage: bash run_pipeline.sh <lang>}
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"42"}          # paper: seed 42; e.g. SEEDS="42 123 456" for variance

if [ "${CRAWL:-0}" = "1" ]; then
  $PY scripts/01_extract_parallel.py --lang "$TGT_LANG" \
      --csv "data/raw/titles/${TGT_LANG}_en_titles_urls.csv" --out-dir data/parallel
fi

$PY scripts/02_project_labels.py --lang "$TGT_LANG" \
    --input "data/parallel/$TGT_LANG/parallel_${TGT_LANG}_en.jsonl" --output-dir "data/projected/$TGT_LANG"

$PY scripts/03_filter_silver.py \
    --input  "data/projected/$TGT_LANG/silver_projected_$TGT_LANG.jsonl" \
    --output "data/projected/$TGT_LANG/silver_filtered_$TGT_LANG.jsonl"

$PY scripts/04_merge_silver_gold.py --lang "$TGT_LANG" \
    --silver "data/projected/$TGT_LANG/silver_filtered_$TGT_LANG.jsonl" \
    --train "data/naamapadam/$TGT_LANG/train.jsonl" --dev "data/naamapadam/$TGT_LANG/dev.jsonl" \
    --test  "data/naamapadam/$TGT_LANG/test.jsonl"  --out-dir "data/augmented/$TGT_LANG"

for MODEL in mbert xlmr; do
  MP=$([ "$MODEL" = mbert ] && echo bert-base-multilingual-cased || echo xlm-roberta-base)
  for SEED in $SEEDS; do
    for SETTING in gold augmented; do
      if [ "$SETTING" = augmented ]; then
        TR=data/augmented/$TGT_LANG/train_aug_$TGT_LANG.jsonl; DV=data/augmented/$TGT_LANG/dev_aug_$TGT_LANG.jsonl
      else
        TR=data/naamapadam/$TGT_LANG/train.jsonl;          DV=data/naamapadam/$TGT_LANG/dev.jsonl
      fi
      $PY scripts/05_train_ner.py --lang "$TGT_LANG" --model-path "$MP" \
          --train "$TR" --dev "$DV" --test "data/naamapadam/$TGT_LANG/test.jsonl" \
          --lrs 1e-5 1e-6 3e-5 3e-6 5e-5 5e-6 --seed "$SEED" --output-dir "runs/$MODEL/$SETTING/$TGT_LANG/seed$SEED"
    done
  done
done

$PY scripts/07_aggregate_results.py --runs-dir runs --out-dir results
