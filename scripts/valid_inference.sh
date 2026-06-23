#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/config.sh"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a DeepSpeed checkpoint directory}"

SIRA_DATA_ROOT="${DATA_ROOT}" \
TORCH_NCCL_BLOCKING_WAIT=1 \
TOKENIZERS_PARALLELISM=True \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
deepspeed --master_port="${MASTER_PORT}" valid_inference.py \
   --version "${CHATUNIVI_MODEL_PATH}" \
   --vision-tower "${CLIP_MODEL_PATH}" \
   --vision_pretrained "${SAM2_CHECKPOINT}" \
   --dataset "reason_seg_sar" \
   --reason_seg_data "SurgRS|surgrs_train.json" \
   --resume "${CHECKPOINT_PATH}" \
   --sample_rates "1" \
   --exp_name "${EXP_NAME}" \
   --balance_sample \
   --grad_accumulation_steps 1 \
   --batch_size 1 \
   --steps_per_epoch 1625 \
   --alpha 0.1 \
   --val_dataset "SurgRS|surgrs_valid.json" \
   --class_meta_json "${DATA_ROOT}/SurgRS/instance_classes.json" \
   --vis_save_path="${OUTPUT_DIR}/sira_inference" \
   "$@"
