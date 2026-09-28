#!/bin/bash
# Stage 5 (fine-tuning). Paper setting = one task (seed 42), 6-LR sweep
# with the LR chosen on dev:
#   sbatch --array=0 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=augmented slurm/train.slurm
# Extra seeds for a variance estimate:
#   sbatch --array=0-4 --export=ALL,TGT_LANG=mr,MODEL=xlmr,SETTING=augmented slurm/train.slurm
#   MODEL   : mbert | xlmr
#   SETTING : gold (Naamapadam only) | augmented (gold + silver)
#SBATCH --job-name=naamasetu-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
: "${TGT_LANG:?}" "${MODEL:?}" "${SETTING:?}"
PYTHON=${PYTHON:-python}
SEEDS=(42 123 456 789 2024)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}
LRS=${LRS:-"1e-5 1e-6 3e-5 3e-6 5e-5 5e-6"}
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

case "$MODEL" in
  mbert) MODEL_PATH=${MODEL_PATH:-bert-base-multilingual-cased} ;;
  xlmr)  MODEL_PATH=${MODEL_PATH:-xlm-roberta-base} ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

if [ "$SETTING" = "augmented" ]; then
  TRAIN=data/augmented/$TGT_LANG/train_aug_$TGT_LANG.jsonl
  DEV=data/augmented/$TGT_LANG/dev_aug_$TGT_LANG.jsonl
else
  TRAIN=data/naamapadam/$TGT_LANG/train.jsonl
  DEV=data/naamapadam/$TGT_LANG/dev.jsonl
fi
TEST=data/naamapadam/$TGT_LANG/test.jsonl

$PYTHON scripts/05_train_ner.py --lang "$TGT_LANG" --model-path "$MODEL_PATH" \
    --train "$TRAIN" --dev "$DEV" --test "$TEST" \
    --lrs $LRS --seed "$SEED" \
    --output-dir "runs/$MODEL/$SETTING/$TGT_LANG/seed$SEED"
