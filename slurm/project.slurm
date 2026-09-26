#!/bin/bash
# Stage 2 (label projection) for one language.
#   sbatch --export=ALL,TGT_LANG=mr slurm/project.slurm
# Edit the #SBATCH lines for your cluster (partition/account/GPU type).
#SBATCH --job-name=naamasetu-proj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${TGT_LANG:?set TGT_LANG, e.g. --export=ALL,TGT_LANG=mr}"
# Call the environment's interpreter directly: `conda activate` is unreliable
# inside batch jobs on some clusters.
PYTHON=${PYTHON:-python}
ENCODER=${ENCODER:-bert-base-multilingual-cased}   # local path works too (add --offline)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

$PYTHON scripts/02_project_labels.py --lang "$TGT_LANG" \
    --input  "data/parallel/$TGT_LANG/parallel_${TGT_LANG}_en.jsonl" \
    --output-dir "data/projected/$TGT_LANG" \
    --model-path "$ENCODER" ${OFFLINE:+--offline}
