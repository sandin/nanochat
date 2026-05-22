#!/bin/bash

# Single-GPU variant of runs/speedrun.sh for one H100 80GB.
# It keeps the same d24 / target-param-data-ratio=8 recipe, but runs without
# torchrun and uses a smaller per-device batch size by default.
#
# Run as:
# bash runs/speedrun_1xh100.sh
#
# Optional knobs:
# DEVICE_BATCH_SIZE=16 bash runs/speedrun_1xh100.sh  # faster if it fits
# DEPTH=22 bash runs/speedrun_1xh100.sh              # smaller/faster model

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

python -m scripts.base_train \
    --depth="$DEPTH" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --fp8 \
    --run="$WANDB_RUN"

python -m scripts.base_eval --device-batch-size="$DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# SFT

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"

python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# Generate report

python -m nanochat.report generate
