#!/usr/bin/env bash

# Edit these paths for the local environment.
DATA_ROOT="${DATA_ROOT:-./data}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
CHATUNIVI_MODEL_PATH="${CHATUNIVI_MODEL_PATH:-${CHECKPOINT_ROOT}/chat-univi}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${CHECKPOINT_ROOT}/clip-vit-large-patch14}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CHECKPOINT_ROOT}/sam2_hiera_large.pt}"

# Shared root for training outputs and inference visualizations.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-24998}"
EXP_NAME="${EXP_NAME:-sira}"
