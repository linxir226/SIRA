# SIRA

Official implementation of **SIRA: Reasoning-Aware Surgical Instrument Segmentation via Query-Anchored Alignment**.

Surgical instrument segmentation (SIS) plays a critical role in robotic assistance and surgical workflow analysis. However, most existing SIS methods formulate segmentation as a category-driven localization problem, limiting their ability to capture procedural context and task-dependent semantics in surgical workflows. We introduce Reasoning-Aware Surgical Instrument Segmentation (RA-SIS), a task formulation that frames segmentation as query-conditioned inference under surgical context. To benchmark this setting, we construct SurgRS, a surgical reasoning segmentation dataset consisting of 41,000 image-text pairs, which aligns instance-level masks with structured query-answer supervision to enable semantic grounding at the pixel level. Based on SurgRS, we propose Surgical Instrument Reasoning and Segmentation Assistant (SIRA), a multimodal framework that disentangles target-level and query-level semantics and integrates them with visual features through query-anchored dual alignment. By aligning query semantics with spatial features and segmentation prompts, SIRA enhances semantic-visual consistency in mask prediction. Extensive experiments on SurgRS demonstrate improvements over existing reasoning-aware baselines.

## Overview

<p align="center">
  <img src="assets/2model.png" alt="Overview of the SIRA framework" width="100%">
</p>

## Installation

We use Python 3.11, PyTorch 2.6.0, and CUDA 12.4.

```bash
conda create -n sira python=3.11 -y
conda activate sira

git clone https://github.com/linxir226/SIRA.git
cd SIRA

# Install PyTorch matching the local CUDA version first. For CUDA 12.4:(you can follow the instructions [here](https://pytorch.org/get-started/locally/))
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e . --no-deps
```

`pip install -e .` compiles the SAM 2 CUDA extension. A working CUDA toolkit and compiler are therefore required.

## Pretrained Models

SIRA requires three upstream pretrained components.

| Component | Configuration key | Purpose |
| --- | --- | --- |
| [Chat-UniVi-7B](https://huggingface.co/Chat-UniVi/Chat-UniVi) | `CHATUNIVI_MODEL_PATH` | Multimodal language model initialization |
| [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | `CLIP_MODEL_PATH` | Chat-UniVi visual encoder |
| [SAM 2 Hiera Large](https://github.com/facebookresearch/sam2) | `SAM2_CHECKPOINT` | Image encoder and mask decoder initialization |

Place the upstream models under `checkpoints/`:

```text
checkpoints/
├── chat-univi/
├── clip-vit-large-patch14/
└── sam2_hiera_large.pt
```

Alternative locations can be set in `scripts/config.sh` or passed through the corresponding environment variables.

## Dataset

SurgRS is available on Hugging Face: [linxir226/SurgRS](https://huggingface.co/datasets/linxir226/SurgRS).

Set `DATA_ROOT` in `scripts/config.sh` to the directory containing SurgRS:

```bash
DATA_ROOT="/path/to/datasets"
```

The default SurgRS layout expected by the provided scripts is:

```text
$DATA_ROOT/
└── SurgRS/
    ├── instance_classes.json
    ├── surgrs_train.json
    ├── surgrs_valid.json
    ├── surgrs_valid_classified.json
    ├── train/
    └── valid/
```

The datasets and generated annotations are not included in this repository. Users must follow the licenses and access terms of the source datasets.

## Training

```bash
DATA_ROOT=/path/to/datasets OUTPUT_DIR=./outputs bash scripts/train.sh
```

GPU IDs, port, experiment name, and output root are configured in `scripts/config.sh`. The default configuration uses GPUs 0 and 1 and writes training outputs to `./outputs/sira`. A single-GPU run can be launched without editing the script:

```bash
GPU_IDS=0 DATA_ROOT=/path/to/datasets OUTPUT_DIR=./outputs bash scripts/train.sh
```

The reported experiments use two RTX 3090 GPUs, a per-GPU batch size of 1, and no gradient accumulation. The script performs 6,500 distributed optimizer steps per epoch, corresponding to 13,000 processed samples per epoch across both GPUs.

### Training Checkpoints

Training outputs are written under `${OUTPUT_DIR}/${EXP_NAME}`:

```text
outputs/
└── <EXP_NAME>/
    ├── train.log
    ├── events.out.tfevents.*
    ├── meta_log_epoch*_giou*_ciou*_dice*.pth
    └── ckpt_model/
        ├── latest
        └── global_step*/
            ├── mp_rank_00_model_states.pt
            └── *_optim_states.pt
```

`ckpt_model/` is the complete DeepSpeed checkpoint directory. The `meta_log_epoch*.pth` files only record epoch and metric information and cannot be used as model weights.

By default, the training code keeps the checkpoint with the best validation gIoU and replaces the previous `ckpt_model/` directory.

To resume training, pass the complete `ckpt_model/` directory:

```bash
DATA_ROOT=/path/to/datasets \
OUTPUT_DIR=./outputs \
bash scripts/train.sh --resume ./outputs/<EXP_NAME>/ckpt_model
```

## Inference

Trained SIRA checkpoints will be released at [linxir226/SIRA](https://huggingface.co/linxir226/SIRA). Until then, inference requires a checkpoint produced by the training script.

After release, place the checkpoint directory under `checkpoints/sira/`:

```text
checkpoints/
└── sira/
    └── ckpt_model/
        ├── latest
        ├── zero_to_fp32.py
        └── global_step*/
```

Standard inference:

```bash
DATA_ROOT=/path/to/datasets \
CHECKPOINT_PATH=./checkpoints/sira/ckpt_model \
bash scripts/valid_inference.sh
```

`CHECKPOINT_PATH` must point to the DeepSpeed checkpoint directory containing `latest`, rather than to `global_step*`, `meta_log_epoch*.pth`, or an individual `.pt` file. 

Inference with metrics grouped by reasoning-query type:

```bash
DATA_ROOT=/path/to/datasets \
CHECKPOINT_PATH=./checkpoints/sira/ckpt_model \
bash scripts/valid_inference_classes.sh
```

## Citation

```bibtex
@misc{zhang2026sira,
  title={SIRA: Reasoning-Aware Surgical Instrument Segmentation via Query-Anchored Alignment},
  author={Zhang, Zhibo and Wang, Qijie and Yan, Zengqiang},
  year={2026},
  url={https://github.com/linxir226/SIRA}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work is built upon [VRS-HQ](https://github.com/SitongGong/VRS-HQ), [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi), [VISA](https://github.com/cilinyan/VISA), and [SAM 2](https://github.com/facebookresearch/sam2). We sincerely thank the authors for their excellent contributions.

The retained or adapted components remain subject to their respective licenses. The `sam2/` implementation is derived from SAM 2.
