#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/config.sh"

SIRA_DATA_ROOT="${DATA_ROOT}" \
TORCH_NCCL_BLOCKING_WAIT=1 \
TOKENIZERS_PARALLELISM=True \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
deepspeed --master_port="${MASTER_PORT}" train.py \
   --version "${CHATUNIVI_MODEL_PATH}" \
   --vision-tower "${CLIP_MODEL_PATH}" \
   --vision_pretrained "${SAM2_CHECKPOINT}" \
   --dataset "reason_seg_sar" \
   --reason_seg_data "SurgRS|surgrs_train.json" \
   --sample_rates "1" \
   --log_base_dir "${OUTPUT_DIR}" \
   --exp_name "${EXP_NAME}" \
   --balance_sample \
   --grad_accumulation_steps 1 \
   --batch_size 1 \
   --steps_per_epoch 6500 \
   --alpha 0.1 \
   --val_dataset "SurgRS|surgrs_valid.json" \
   --epochs 10 \
   "$@"
