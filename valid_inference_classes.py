import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from functools import partial
import traceback

import cv2
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont

import re
import math
import numpy as np
from collections import defaultdict, Counter
import torch.distributed as dist

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.SIRA_multiseg import SiraForCausalLM
from model.univi.config import ModelConfig, DataConfig
from model.llava import conversation as conversation_lib      # 为了测试暂时注释这三行代码
from dataset.multi_dataset import HybridDataset, ValDataset, collate_fn
from dataset.rvos_clip_eval_dataset import RVOSEvalDataset
from dataset.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda, intersectionAndUnionGPU
)

def parse_args(args):
    parser = argparse.ArgumentParser(description="SIRA Model Inference")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="./checkpoints/chat-univi"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="./checkpoints/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # sem_seg||refer_seg||vqa||
    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg||chatunivi||rvos", type=str
    )
    # 9,3,3,
    parser.add_argument("--sample_rates", default="9,3,3,1,4,12", type=str)

    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)


    parser.add_argument("--val_dataset", default="mevis_val", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)         # path to save results and checkpoints
    parser.add_argument("--exp_name", default="SIRA", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=1500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    # 梯度累计次数：多少次前向传播后，反向传播一次
    parser.add_argument(
        "--grad_accumulation_steps",
        default=32,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)            # 对学习率进行过调整，原来为0.0003
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=True)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam2_hiera_large.pt", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_memory_modules", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    #  parser.add_argument("--auto_resume", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # Balance 采样
    parser.add_argument('--balance_sample', action='store_true', default=True)

    # ChatUnivi 训练集
    parser.add_argument('--univi_data_list', default="mimic||sqa||video", type=str)
    parser.add_argument('--univi_data_ratio', default="1||1||1", type=str)
    parser.add_argument('--univi_sample_frame_range', default="8,12", type=str)
    parser.add_argument('--univi_max_image_len', default=64, type=int)  # no use
    #  ||davis17_train
    parser.add_argument("--rvos_seg_data", default="mevis_train||refytvos_train||davis17_train||revos_train||lvvis_train", type=str)
    parser.add_argument('--rvos_sample_ratio', default="4000||15000||400||3000||3000", type=str,)
    parser.add_argument('--rvos_num_frames_sample_range', default='8,12', type=str)
    parser.add_argument('--rvos_sample_policy', default='uniform', type=str)
    parser.add_argument('--rvos_max_image_token', type=int, default=12)  # 验证集中，最多选用几帧

    # 只需要一个帧分割token，有不同的嵌入
    parser.add_argument("--num_seg_token", default=1, type=int)

    parser.add_argument("--alpha", default=0.1, type=float)

    parser.add_argument(
        "--class_meta_json",
        default="./data/SurgRS/instance_classes.json",
        type=str,
    )
    parser.add_argument(
        "--classified_json",
        default="./data/SurgRS/surgrs_valid_classified.json",
        type=str,
    )
    parser.add_argument("--vis_alpha", default=0.5, type=float)
    parser.add_argument("--vis_rank_subdir", action="store_true", default=True)
    parser.add_argument("--vis_enable", action="store_true", default=False)
    parser.add_argument("--vis_save_all_frames", action="store_true", default=False)  # True=序列每帧都存；False=只存第一帧

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    ##################################################### 日志
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0: # 0, 1, 2, 3, 4, 5, 6, 7
        os.makedirs(args.log_dir, exist_ok=True)
        # 定义日志存放路径
        log_path = os.path.join(args.log_dir, "test.log")

        # 配置全局 logging , 可在任意文件直接使用 logging.info(), 需要import logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),  # 输出到文件
                logging.StreamHandler(sys.stdout),  # 同时输出到控制台
            ],
        )
    else:
        writer = None
    logging.info("Arguments: " + str(args))

    ##################################################### 创建标记嵌入模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # args.version,
        pretrained_model_name_or_path=args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # SIRA special tokens: target token and a fixed block of query tokens.
    if args.num_seg_token == 1:
        num_added_tokens = tokenizer.add_tokens("<SEG>", special_tokens=True)
        args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["<SEG{}>".format(i) for i in range(args.num_seg_token)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer("<SEG{}>".format(i), add_special_tokens=False).input_ids[0] for i in range(args.num_seg_token)]

    tokenizer.add_tokens("<QUERY>", special_tokens=True)
    args.query_token_idx = tokenizer("<QUERY>", add_special_tokens=False).input_ids[0]

    # 是否使用图像起始与结束标记，这里默认不使用
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    ##################################################### Declare the model 总体模型声明
    model_args = {
        "train_memory_modules": args.train_memory_modules, # new, 默认是False
        "train_mask_decoder": args.train_mask_decoder, # train_mask_decoder: True
        "out_dim": args.out_dim, # 256
        "ce_loss_weight": args.ce_loss_weight, # 1.0, auto-regressive binary cross entropy loss for text generation
        "dice_loss_weight": args.dice_loss_weight, # 0.5, segmentation dice loss
        "bce_loss_weight": args.bce_loss_weight, # 2, segmentation bce loss
        "seg_token_idx": args.seg_token_idx, # 分割嵌入索引
        "seg_token_num": args.num_seg_token, # new, 1, 分割嵌入数量
        "query_token_idx": args.query_token_idx,
        "num_query_tokens": 64,
        "vision_pretrained": args.vision_pretrained, # sam2 large checkpoint
        "vision_tower": args.vision_tower, # CLIP vision encoder of chat-univi/LLaVA model
        "alpha": args.alpha,
        "use_mm_start_end": args.use_mm_start_end, # chatunivi是im, 原版llava是mm, 默认是False，正确的名字还是use_mm_start_end，只不过这里换用了im，在里面初始化的时候就使用了默认值true，应该有问题，我先改成use_mm_start_end，比较符合作者意思
        # "local_rank": args.local_rank,
        # "num_classes": args.num_classes,  # 未来多类分割时使用
    }
    # model_args = AutoConfig.from_pretrained(args.version)
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 使用llm预训练方法加载模型，后面补充sam2方法
    model = SiraForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id # 2
    model.config.bos_token_id = tokenizer.bos_token_id # 1
    model.config.pad_token_id = tokenizer.pad_token_id # 0 实际上和预训练值一样的

    # 确保模型允许输入梯度回传，允许梯度检查点
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 加载clip预训练模型  在llm模型中有定义函数  def get_model(self): return self.model
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    # chat-univi模型需要加载聚类
    model_args_from_pt = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.version) # 提示：The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    model_args_from_pt.use_cluster = True
    model_args_from_pt.freeze = False
    model_args_from_pt.mm_tune = True
    model_args_from_pt.spatial_cluster_rate0 = 64
    model_args_from_pt.spatial_cluster_rate1 = 32
    model_args_from_pt.spatial_cluster_rate2 = 16
    model_args_from_pt.temporal_cluster_rate = 0.0625
    model_args_from_pt.use_cluster = True
    model_args_from_pt.vision_tune = False
    model.get_model().initialize_cluster_modules(model_args_from_pt)

    # sam2 没有默认初始化分割头，这里手动判断是否需要初始化
    model.get_model().initialize_lisa_modules(model.get_model().config) # 创建一下sam2和预训练模型架构和[seg]投影头

    # clip不训练, 映射头需要训练
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    ##################################################### 加载对话模板，好像没用到
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    ##################################################### lora创建与参数冻结
    lora_r = args.lora_r
    if lora_r > 0: # 使用lora, 默认秩为8

        # 找到需要训练的lora模块
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls) # 需要是线性层
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ] # 不需要训练的层, 限定目标在语言模型本身
                    )
                    and any([x in name for x in lora_target_modules]) # q_proj,v_proj层(lora)需要训练
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        # 加载Lora权重
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印训练参数 trainable params: 4,194,304 || all params: 7,292,248,370 || trainable%: 0.0575

    model.resize_token_embeddings(len(tokenizer)) # 扩展embedding & lm_head，打印The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.

    # Keep the trainable-module definition consistent with the training entry point.
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in [
                    "lm_head",
                    "embed_tokens",
                    "mask_decoder",
                    "text_hidden_fcs",
                    "query_transformer",
                    "cross_attention",
                ]
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    ##################################################### 多GPU时分布式训练初始化
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    ##################################################### 初始化训练数据集
    val_out_dirname = args.val_dataset if '_split' not in args.val_dataset else args.val_dataset.split('_split', 1)[0]
    val_dataset = ValDataset(
        tokenizer                = tokenizer,
        vision_tower             = args.vision_tower,
        image_size               = args.image_size,
        val_dataset              = args.val_dataset,
        use_sira_tokens          = True,
        )

    ##################################################### 模型加速, deepspeed配置
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
    }
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    ##################################################### 加载模型存档点, resume deepspeed checkpoint
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        logging.info(
            "loaded checkpoint from {}, global step {}, inferred start epoch {}".format(
                args.resume, ckpt_dir.replace("global_step", ""), args.start_epoch
            )
        )


    ##################################################### 验证数据集的load无法加速，单独加载 validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    ##################################################### 开始训练与验证
    id2bgr, id2title, id_order = load_instance_classes(args.class_meta_json)
    args._id2bgr = id2bgr
    args._id2title = id2title
    args._id_order = id_order

    result = validate_cell_type(val_loader, model_engine, args)


def validate_cell_type(val_loader, model_engine, args):
    img2qtypes, qid2name = build_qtype_index(args.classified_json)
    overall_intersection = AverageMeter(
        "Overall_Intersection", fmt=":6.3f", summary_type=Summary.SUM
    )
    overall_union = AverageMeter("Overall_Union", fmt=":6.3f", summary_type=Summary.SUM)
    overall_iou = AverageMeter("Overall_IoU", fmt=":6.3f", summary_type=Summary.AVERAGE)
    dice_meter = AverageMeter("Dice", fmt=":6.3f", summary_type=Summary.AVERAGE)
    bleu_meter = AverageMeter("BLEU4", fmt=":6.3f", summary_type=Summary.AVERAGE)
    f1_meter = AverageMeter("TextF1", fmt=":6.3f", summary_type=Summary.AVERAGE)

    # 逐样本落盘（rank0）
    per_sample_rows = []

    model_engine.eval()
    category_metrics = {}
    # --- 新增：按 qtype 统计 ---
    qtype_metrics = {}
    # --- 新增：按 (qtype, cat) 统计 ---
    qtype_category_metrics = {}

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        category_ids = input_dict["category_ids"]

        # cast
        if args.precision == "fp16":
            input_dict["images"] = [i.half() for i in input_dict["images"]]
            input_dict["images_clip"] = [i.half() for i in input_dict["images_clip"]]
        elif args.precision == "bf16":
            input_dict["images"] = [i.bfloat16() for i in input_dict["images"]]
            input_dict["images_clip"] = [i.bfloat16() for i in input_dict["images_clip"]]
        else:
            input_dict["images"] = [i.float() for i in input_dict["images"]]
            input_dict["images_clip"] = [i.float() for i in input_dict["images_clip"]]

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # 读取必要字段（都在 input_dict 里）
        img_path, questions, answers, sampled_classes, focus_ids_per_q = extract_qa_from_inputdict(input_dict)
        # --------- prediction & gt ---------
        pred_masks = output_dict["pred_masks"]   # list length=1 most cases
        if isinstance(pred_masks, list):
            pred_masks = pred_masks[0]
        if pred_masks.dim() == 4:  # [N,1,H,W]
            pred_masks = pred_masks[:, 0]
        output_list = (pred_masks > 0).int()      # [N,H,W]

        masks_list = output_dict["gt_masks"][0].int()  # [N,H,W]
        assert output_list.shape == masks_list.shape
        if output_list.numel() == 0:
            continue
        category_ids = flatten_category_ids(category_ids[0])

        # ============================
        # Visualization (per-sample)
        # ============================
        if args.vis_enable and args.local_rank == 0:
            try:
                image_bgr = read_image_bgr(img_path)

                if image_bgr is None:
                    # 如果 dataset 没给路径或者路径读不到，就不保存（但不影响评测）
                    if args.local_rank == 0:
                        logging.warning(f"[VIS] skip: cannot read image path: {img_path}")
                else:
                    # 文件名：尽量稳定且不冲突
                    # 如果路径存在，用文件名；否则用计数
                    if img_path is not None:
                        base = os.path.splitext(os.path.basename(img_path))[0]
                    else:
                        base = f"sample_{int(time.time()*1000)}"

                    # 分布式避免冲突：每个 rank 单独子目录（推荐）
                    video_id = guess_video_id(img_path)  # 你之前加的那个函数
                    frame_folder = base                    # base 就是图片名(不含后缀)

                    # ------- 1) 每张图一个文件夹 -------
                    if getattr(args, "vis_rank_subdir", True):
                        save_root = os.path.join(args.vis_save_path, f"rank{args.local_rank}", video_id, frame_folder)
                    else:
                        save_root = os.path.join(args.vis_save_path, video_id, frame_folder)
                    # ------- 2) 保存整图 all/ + original -------
                    save_all_bundle_only(
                        save_root=save_root,
                        image_bgr=image_bgr,
                        pred_masks_NHW=output_list,
                        gt_masks_NHW=masks_list,
                        category_ids_N=category_ids,
                        id2bgr=args._id2bgr,
                        id2title=args._id2title,
                        alpha=getattr(args, "vis_alpha", 0.5),
                    )

                    # ------- 3) 问题 all/ + meta.json -------
                    img_meta = {
                        "image_path": img_path,
                        "instance_category_ids": [int(x) for x in category_ids],
                        "qa": [
                            {
                                "q_index": i + 1,
                                "question": questions[i],
                                "sampled_class": sampled_classes[i] if i < len(sampled_classes) else "",
                                "answer": answers[i] if i < len(answers) else "",
                                "focus_category_ids": focus_ids_per_q[i] if i < len(focus_ids_per_q) else [],
                            }
                            for i in range(len(questions))
                        ],
                    }
                    os.makedirs(save_root, exist_ok=True)
                    with open(os.path.join(save_root, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(img_meta, f, ensure_ascii=False, indent=2)

                    # ------- 4) 每题一个文件夹 q001/q002... -------
                    for i in range(len(questions)):
                        q_idx = i + 1
                        q_dir = os.path.join(save_root, f"q{q_idx:03d}")
                        focus_ids = focus_ids_per_q[i] if i < len(focus_ids_per_q) else []

                        pred_sel, gt_sel, cid_sel = pick_masks_for_question_by_cids(
                            pred_masks_NHW=output_list,
                            gt_masks_NHW=masks_list,
                            category_ids_N=category_ids,
                            focus_ids=focus_ids
                        )

                        # 保存该题主输出 + by_class + meta.json（用我上次给你的 save_question_bundle）
                        save_question_bundle(
                            q_dir=q_dir,
                            q_idx=q_idx,
                            question=questions[i],
                            answer=answers[i] if i < len(answers) else "",
                            focus_ids=focus_ids,
                            image_bgr=image_bgr,
                            pred_masks_sel=pred_sel,
                            gt_masks_sel=gt_sel,
                            pred_cids_sel=cid_sel,
                            gt_cids_sel=cid_sel,
                            id2bgr=args._id2bgr,
                            id2title=args._id2title,
                            alpha=getattr(args, "vis_alpha", 0.5),
                        )

                        # 题级 meta.json 里建议额外写 sampled_class，便于回溯
                        q_meta = {
                            "q_index": q_idx,
                            "question": questions[i],
                            "sampled_class": sampled_classes[i],
                            "answer": answers[i] if i < len(answers) else "",
                            "focus_category_ids": focus_ids,
                            "image_path": img_path,
                        }
                        with open(os.path.join(q_dir, "meta.json"), "w", encoding="utf-8") as f:
                            json.dump(q_meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                if args.local_rank == 0:
                    logging.warning(f"[VIS] exception: {e}")
                    logging.warning(traceback.format_exc())


        ### 安全检查
        if args.local_rank == 0:
            exp = sum(len(x) for x in (focus_ids_per_q or []))
            got = len(output_list)
            if exp != got:
                logging.warning(f"[ALIGN WARN] sum(len(focus_ids_per_q))={exp} != num_masks={got} for {rel_img}")


        ### 文本与保存
        rel_img = normalize_img_path(img_path)
        qtype_ids_for_img = img2qtypes.get(rel_img, None)  # list[int] or None
        # 用于把 mask 索引映射到 question 索引
        question_idx = 0
        remain_in_question = (len(focus_ids_per_q[0]) if focus_ids_per_q and len(focus_ids_per_q) > 0 else 0)
        last_qidx = -1
        if len(output_list) == 0:
            continue
        for mask_i, output_i, cat in zip(masks_list, output_list, category_ids):
            intersection, union, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            fg_intersection = intersection[1].item()
            fg_union = union[1].item()
            # 当并集为 0（空目标）时定义 IoU 为 1.0
            fg_iou = 1.0 if fg_union == 0 else fg_intersection / (fg_union + 1e-5)

            overall_intersection.update(fg_intersection, n=1)
            overall_union.update(fg_union, n=1)
            overall_iou.update(fg_iou, n=1)

            cat_id = int(cat.item()) if torch.is_tensor(cat) else int(cat)
            if cat_id not in category_metrics:
                category_metrics[cat_id] = {
                    "count": AverageMeter(
                        f"Count_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "intersection": AverageMeter(
                        f"Intersection_cat_{cat_id}",
                        fmt=":6.3f",
                        summary_type=Summary.SUM,
                    ),
                    "union": AverageMeter(
                        f"Union_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "iou": AverageMeter(
                        f"IoU_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                    "dice": AverageMeter(
                        f"Dice_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                }

            category_metrics[cat_id]["count"].update(1.0, n=1)
            category_metrics[cat_id]["intersection"].update(fg_intersection, n=1)
            category_metrics[cat_id]["union"].update(fg_union, n=1)
            category_metrics[cat_id]["iou"].update(fg_iou, n=1)

            d = dice_fg(output_i, mask_i, ignore_index=255)
            if d is not None:
                dice_meter.update(d.item(), n=1)
                category_metrics[cat_id]["dice"].update(d.item(), n=1)

            # ================= 修正：从 mask -> question 对齐 qtype =================
            if qtype_ids_for_img is not None and question_idx < len(qtype_ids_for_img):
                qid = int(qtype_ids_for_img[question_idx])
            else:
                qid = -1

            # (A) qtype 级别
            if qid not in qtype_metrics:
                qtype_metrics[qid] = {
                    "count_question": AverageMeter(
                        f"Count_question_qtype_{qid}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "count": AverageMeter(
                        f"Count_qtype_{qid}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "intersection": AverageMeter(
                        f"Intersection_qtype_{qid}",
                        fmt=":6.3f",
                        summary_type=Summary.SUM,
                    ),
                    "union": AverageMeter(
                        f"Union_qtype_{qid}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "iou": AverageMeter(
                        f"IoU_qtype_{qid}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                    "dice": AverageMeter(
                        f"Dice_qtype_{qid}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                }

            if question_idx != last_qidx:
                qtype_metrics[qid]["count_question"].update(1.0, n=1)
                last_qidx = question_idx
            qtype_metrics[qid]["count"].update(1.0, n=1)
            qtype_metrics[qid]["intersection"].update(fg_intersection, n=1)
            qtype_metrics[qid]["union"].update(fg_union, n=1)
            qtype_metrics[qid]["iou"].update(fg_iou, n=1)
            if d is not None:
                qtype_metrics[qid]["dice"].update(d.item(), n=1)

            # (B) (qtype, cat) 级别
            key = (qid, cat_id)
            if key not in qtype_category_metrics:
                qtype_category_metrics[key] = {
                    "count": AverageMeter(
                        f"Count_q{qid}_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "intersection": AverageMeter(
                        f"Intersection_q{qid}_cat_{cat_id}",
                        fmt=":6.3f",
                        summary_type=Summary.SUM,
                    ),
                    "union": AverageMeter(
                        f"Union_q{qid}_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "iou": AverageMeter(
                        f"IoU_q{qid}_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                    "dice": AverageMeter(
                        f"Dice_q{qid}_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                }

            qtype_category_metrics[key]["count"].update(1.0, n=1)
            qtype_category_metrics[key]["intersection"].update(fg_intersection, n=1)
            qtype_category_metrics[key]["union"].update(fg_union, n=1)
            qtype_category_metrics[key]["iou"].update(fg_iou, n=1)
            if d is not None:
                qtype_category_metrics[key]["dice"].update(d.item(), n=1)

            # 当前问题还剩多少 mask
            remain_in_question -= 1
            if remain_in_question <= 0:
                question_idx += 1
                if focus_ids_per_q and question_idx < len(focus_ids_per_q):
                    remain_in_question = len(focus_ids_per_q[question_idx])
                else:
                    remain_in_question = 0

        # ---------- text metrics (if available) ----------
        gt_text, pred_text = extract_text_gt_pred(input_dict, output_dict)
        if gt_text is not None and pred_text is not None:
            b4 = bleu4(pred_text, gt_text, smooth=True)
            _, _, f1 = token_f1(pred_text, gt_text)
            bleu_meter.update(b4, n=1)
            f1_meter.update(f1, n=1)

            # 逐样本记录（建议用图片名做主键）
            img_key = img_path if img_path else f"idx_{len(per_sample_rows)}"
            per_sample_rows.append({
                "image": img_key, # 应配合video
                "bleu4": b4,
                "text_f1": f1,
                "gt_text": gt_text,
                "pred_text": pred_text,
            })

        # shape [2, 1]
        # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

    if dist.is_initialized():
        for cat_id in category_metrics:
            for meter in category_metrics[cat_id].values():
                meter.all_reduce()
        overall_intersection.all_reduce()
        overall_union.all_reduce()
        overall_iou.all_reduce()
        dice_meter.all_reduce()
        bleu_meter.all_reduce()
        f1_meter.all_reduce()
        # --- 新增：all_reduce qtype 与 (qtype,cat) ---
        for qid in qtype_metrics:
            for meter in qtype_metrics[qid].values():
                meter.all_reduce()
        for key in qtype_category_metrics:
            for meter in qtype_category_metrics[key].values():
                meter.all_reduce()


    results = {}
    for cat_id, meters in category_metrics.items():
        inter_sum = meters["intersection"].sum
        union_sum = meters["union"].sum
        ciou = 1.0 if union_sum == 0 else inter_sum / (union_sum + 1e-10)
        giou = meters["iou"].avg
        dice = meters["dice"].avg
        count = int(round(meters["count"].sum))
        results[cat_id] = {"count": count, "ciou": ciou, "giou": giou, "dice": dice}
        if args.local_rank == 0:
            logging.info(f"Category {cat_id}: count: {count}, ciou: {ciou:.4f}, giou: {giou:.4f}, dice: {dice:.4f}")

    # --- 新增：汇总 qtype_results ---
    qtype_results = {}
    for qid, meters in qtype_metrics.items():
        inter_sum = meters["intersection"].sum
        union_sum = meters["union"].sum
        ciou = 1.0 if union_sum == 0 else inter_sum / (union_sum + 1e-10)
        giou = meters["iou"].avg
        dice = meters["dice"].avg
        count = int(round(meters["count"].sum))
        count_question = int(round(meters["count_question"].sum))
        qname = qid2name.get(qid, "UNKNOWN") if qid >= 0 else "UNKNOWN"
        qtype_results[qid] = {"name": qname, "count_question": count_question, "count": count, "ciou": ciou, "giou": giou, "dice": dice}
        if args.local_rank == 0:
            logging.info(f"QType {qid} ({qname}): count_question: {count_question}, count: {count}, ciou: {ciou:.4f}, giou: {giou:.4f}, dice: {dice:.4f}")

    overall_ciou = (
        1.0
        if overall_union.sum == 0
        else overall_intersection.sum / (overall_union.sum + 1e-10)
    )
    overall_giou = overall_iou.avg

    if args.local_rank == 0:
        args.val_json_name = args.val_dataset.split("|")[-1]
        csv_file = args.val_json_name.replace("json", "csv")
        csv_file = f"{args.log_dir}/{csv_file}"
        # 将类别 id 按升序排列
        sorted_cat_ids = sorted(results.keys())
        header = (
            ["metric"] + [f"cat_{cat_id}" for cat_id in sorted_cat_ids] + ["overall"]
        )
        with open(csv_file, mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            ciou_row = (
                ["ciou"]
                + [results[cat_id]["ciou"] for cat_id in sorted_cat_ids]
                + [overall_ciou]
            )
            csv_writer.writerow(ciou_row)
            giou_row = (
                ["giou"]
                + [results[cat_id]["giou"] for cat_id in sorted_cat_ids]
                + [overall_giou]
            )
            csv_writer.writerow(giou_row)
            dice_row = (
                ["dice"]
                + [results[cat_id]["dice"] for cat_id in sorted_cat_ids]
                + [dice_meter.avg]
            )
            csv_writer.writerow(dice_row)
            logging.info(f"Results saved to {csv_file}")

    if args.local_rank == 0:
        # 1) 保存整体指标
        metrics_file = args.val_json_name.replace("json", "metrics.json")
        metrics_file = f"{args.log_dir}/{metrics_file}"
        metrics = {
            "overall_ciou": float(overall_ciou),
            "overall_giou": float(overall_giou),
            "dice": float(dice_meter.avg),
            "bleu4": float(bleu_meter.avg),
            "text_f1": float(f1_meter.avg),
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logging.info(f"Overall metrics saved to {metrics_file}")

        # 2) 保存逐样本（JSONL）
        jsonl_file = args.val_json_name.replace("json", "per_sample.jsonl")
        jsonl_file = f"{args.log_dir}/{jsonl_file}"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for r in per_sample_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info(f"Per-sample text metrics saved to {jsonl_file}")

        # 3) 也可另存一个简洁 CSV（只含分数）
        csv2_file = args.val_json_name.replace("json", "text_scores.csv")
        csv2_file = f"{args.log_dir}/{csv2_file}"
        with open(csv2_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "bleu4", "text_f1"])
            for r in per_sample_rows:
                w.writerow([r["image"], r["bleu4"], r["text_f1"]])
        logging.info(f"Text score CSV saved to {csv2_file}")

        # --- 新增：保存 qtype_summary.csv ---
        qtype_csv = args.val_json_name.replace("json", "qtype_summary.csv")
        qtype_csv = f"{args.log_dir}/{qtype_csv}"
        with open(qtype_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qtype_id", "qtype_name", "count", "count_question", "ciou", "giou", "dice"])
            for qid in sorted(qtype_results.keys()):
                r = qtype_results[qid]
                w.writerow([qid, r["name"], r["count"], r["count_question"], r["ciou"], r["giou"], r["dice"]])
        logging.info(f"QType summary saved to {qtype_csv}")

        # --- 新增：保存 qtype_cat_summary.csv ---
        qtype_cat_csv = args.val_json_name.replace("json", "qtype_cat_summary.csv")
        qtype_cat_csv = f"{args.log_dir}/{qtype_cat_csv}"
        with open(qtype_cat_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qtype_id", "qtype_name", "cat_id", "count", "ciou", "giou", "dice"])
            # 逐 key 汇总
            rows = []
            for (qid, cat_id), meters in qtype_category_metrics.items():
                inter_sum = meters["intersection"].sum
                union_sum = meters["union"].sum
                ciou = 1.0 if union_sum == 0 else inter_sum / (union_sum + 1e-10)
                giou = meters["iou"].avg
                dice = meters["dice"].avg
                count = int(round(meters["count"].sum))
                qname = qid2name.get(qid, "UNKNOWN") if qid >= 0 else "UNKNOWN"
                rows.append((qid, cat_id, qname, count, ciou, giou, dice))
            for qid, cat_id, qname, count, ciou, giou, dice in sorted(rows, key=lambda x: (x[0], x[1])):
                w.writerow([qid, qname, cat_id, count, ciou, giou, dice])
        logging.info(f"QType+Cat summary saved to {qtype_cat_csv}")

    results["overall"] = {"ciou": overall_ciou, "giou": overall_giou}
    results["overall"].update({
        "dice": float(dice_meter.avg),
        "bleu4": float(bleu_meter.avg),
        "text_f1": float(f1_meter.avg),
    })
    results["qtype"] = qtype_results
    # results["qtype_target"] = qtype_category_metrics

    return results


def build_qtype_index(classified_json_path):
    with open(classified_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    img2qtypes = {}
    for item in meta.get("data", []):
        rel = str(item["image_path"]).replace("\\", "/")
        img2qtypes[rel] = item.get("question_type_ids", None)
    qtype_dict = meta.get("question_type_dict", {})
    qid2name = {int(k): v.get("name", str(k)) for k, v in qtype_dict.items()}
    return img2qtypes, qid2name

def normalize_img_path(img_path):
    """
    更稳一点：优先截取 split/video_xx/rgb/xxx.png
    """
    if img_path is None:
        return None
    p = str(img_path).replace("\\", "/")
    # 尝试截取 valid/xxx 或 train/xxx
    if "/valid/" in p:
        return "valid/" + p.split("/valid/", 1)[1]
    if "/train/" in p:
        return "train/" + p.split("/train/", 1)[1]
    if p.startswith("valid/") or p.startswith("train/"):
        return p
    # 兜底：取最后四段
    parts = p.split("/")
    if len(parts) >= 4:
        return "/".join(parts[-4:])
    return p


def load_instance_classes(json_path: str):
    """
    返回:
      id2rgb: {id: (b,g,r)}  # 注意cv2是BGR
      id2title: {id: title}
      id_order: [id1,id2,...]  # 按json里的顺序
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta.get("classes", [])
    id2rgb = {}
    id2title = {}
    id_order = []
    for c in classes:
        cid = int(c["id"])
        rgb = c.get("color_rgb", [255, 255, 255])
        # cv2 uses BGR
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        id2rgb[cid] = bgr
        id2title[cid] = str(c.get("title", f"class_{cid}"))
        id_order.append(cid)
    return id2rgb, id2title, id_order


def flatten_category_ids(values):
    if values is None:
        return []
    if torch.is_tensor(values):
        if values.numel() == 0:
            return []
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    if isinstance(values, np.ndarray):
        return [int(x) for x in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        out = []
        for item in values:
            out.extend(flatten_category_ids(item))
        return out
    return [int(values)]


def extract_qa_from_inputdict(input_dict):
    """
    适配你给出的真实结构（batch=1）：
      image_paths: [str]
      questions_list: [[q1,q2,...]]
      sampled_classes_list: [[s1,s2,...]]
      conversation_list: [conv1, conv2, ...]          # 注意：不是 [[...]]
      category_ids: [[cid0,cid1,...]]                 # 注意：不是 tensor
    返回：
      img_path, questions, answers, sampled_classes, focus_ids_per_q
    """

    # 1) image path
    img_path = None
    if "image_paths" in input_dict and input_dict["image_paths"]:
        img_path = input_dict["image_paths"][0]

    # 2) questions (batch=1 -> 取 [0])
    questions = []
    if "questions_list" in input_dict and input_dict["questions_list"]:
        questions = list(input_dict["questions_list"][0])

    # 3) sampled_classes (batch=1 -> 取 [0])
    sampled_classes = []
    if "sampled_classes_list" in input_dict and input_dict["sampled_classes_list"]:
        sampled_classes = list(input_dict["sampled_classes_list"][0])

    # 4) answers：从 conversation_list 解析（conversation_list 直接是 list[str]）
    convs = input_dict.get("conversation_list", [])
    answers = []
    for s in convs:
        if s is None:
            answers.append("")
            continue
        s = str(s)
        if "ASSISTANT:" in s:
            ans = s.split("ASSISTANT:", 1)[-1].strip()
        elif "<gpt>:" in s:
            ans = s.split("<gpt>:", 1)[-1].strip()
        else:
            ans = ""
        # 去掉结尾 </s>（可选，但建议）
        ans = ans.replace("</s>", "").strip()
        answers.append(ans)

    # 5) 对齐长度（不再 padding 0，只补空串）
    questions = [("" if q is None else str(q)) for q in questions]
    sampled_classes = [("" if s is None else str(s)) for s in sampled_classes]
    answers = [("" if a is None else str(a)) for a in answers]

    Q = max(len(questions), len(sampled_classes), len(answers))
    while len(questions) < Q: questions.append("")
    while len(sampled_classes) < Q: sampled_classes.append("")
    while len(answers) < Q: answers.append("")

    # 6) category_ids：batch=1 -> 取 [0] 得到 cid 序列（list[int]）
    cids = input_dict.get("category_ids", None)
    if isinstance(cids, (list, tuple)) and len(cids) > 0:
        cid_list = flatten_category_ids(cids[0])
    else:
        cid_list = flatten_category_ids(cids)

    # Split category IDs according to the number of <SEG> tokens per answer.
    focus_ids_per_q = []
    ptr = 0
    for i in range(Q):
        # 用正则更稳一点（防止 [seg] 大小写等）
        n_seg = len(re.findall(r"<SEG>", answers[i])) if answers[i] else 0
        if n_seg <= 0:
            focus_ids_per_q.append([])
            continue

        focus = cid_list[ptr: ptr + n_seg]
        ptr += n_seg
        focus_ids_per_q.append([int(x) for x in focus])

    return img_path, questions, answers, sampled_classes, focus_ids_per_q


def read_image_bgr(path: str):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def guess_video_id(img_path: str):
    if img_path:
        # e.g. /data/.../train/video_01/rgb/00012.png
        parts = img_path.split(os.sep)
        for i in range(len(parts) - 1):
            if parts[i].lower().startswith("video"):
                return parts[i]
        # 兜底：父目录名
        return os.path.basename(os.path.dirname(img_path))
    return "unknown_video"


def save_all_bundle_only(
    save_root: str,
    image_bgr: np.ndarray,
    pred_masks_NHW,
    gt_masks_NHW,
    category_ids_N,
    id2bgr: Dict[int, tuple],
    id2title: Dict[int, str],
    alpha: float = 0.45
):
    os.makedirs(save_root, exist_ok=True)
    all_dir = os.path.join(save_root, "all")
    os.makedirs(all_dir, exist_ok=True)

    H, W = image_bgr.shape[:2]

    # original
    cv2.imwrite(os.path.join(save_root, "original.jpg"), image_bgr)

    # merged color
    pred_color, _, _ = _build_merged_color_mask(pred_masks_NHW, category_ids_N, id2bgr, H, W)
    gt_color,   _, _ = _build_merged_color_mask(gt_masks_NHW,   category_ids_N, id2bgr, H, W)

    # overlay
    pred_overlay = _alpha_overlay(image_bgr, pred_color, alpha=alpha)
    gt_overlay   = _alpha_overlay(image_bgr, gt_color,   alpha=alpha)

    # label
    pred_cls_mask = _build_merged_class_id_mask(pred_masks_NHW, category_ids_N, H, W)
    gt_cls_mask   = _build_merged_class_id_mask(gt_masks_NHW,   category_ids_N, H, W)

    pred_overlay_pil = Image.fromarray(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    gt_overlay_pil   = Image.fromarray(cv2.cvtColor(gt_overlay,   cv2.COLOR_BGR2RGB))
    pred_labeled_pil = _draw_labels_from_mask(pred_overlay_pil, pred_cls_mask, id2title)
    gt_labeled_pil   = _draw_labels_from_mask(gt_overlay_pil,   gt_cls_mask,   id2title)
    pred_overlay_label = cv2.cvtColor(np.array(pred_labeled_pil), cv2.COLOR_RGB2BGR)
    gt_overlay_label   = cv2.cvtColor(np.array(gt_labeled_pil),   cv2.COLOR_RGB2BGR)
    # legend（左上角色块清单）
    # 直接用 category_ids_N 当 present ids（更便宜），或用 merged_color_mask 的 present set 都行
    present_ids = [int(x.item()) if torch.is_tensor(x) else int(x) for x in category_ids_N]
    pred_overlay_label = _draw_legend(pred_overlay_label, present_ids, id2title, id2bgr)
    gt_overlay_label   = _draw_legend(gt_overlay_label,   present_ids, id2title, id2bgr)

    # write
    cv2.imwrite(os.path.join(all_dir, "pred_merged_color.png"), pred_color)
    cv2.imwrite(os.path.join(all_dir, "pred_overlay.jpg"), pred_overlay)
    cv2.imwrite(os.path.join(all_dir, "pred_overlay_label.jpg"), pred_overlay_label)

    cv2.imwrite(os.path.join(all_dir, "gt_merged_color.png"), gt_color)
    cv2.imwrite(os.path.join(all_dir, "gt_overlay.jpg"), gt_overlay)
    cv2.imwrite(os.path.join(all_dir, "gt_overlay_label.jpg"), gt_overlay_label)


def pick_masks_for_question_by_cids(pred_masks_NHW, gt_masks_NHW, category_ids_N, focus_ids):
    """
    pred_masks_NHW / gt_masks_NHW: [N,H,W] (torch)
    category_ids_N: [N] (torch)
    focus_ids: List[int]
    返回该题对应的 masks 子集（可以多个）
    """
    focus = set(int(x) for x in focus_ids)
    pred_sel, gt_sel, cid_sel = [], [], []
    for i, cid in enumerate(category_ids_N):
        cid_int = int(cid.item()) if torch.is_tensor(cid) else int(cid)
        if cid_int in focus:
            pred_sel.append(pred_masks_NHW[i])
            gt_sel.append(gt_masks_NHW[i])
            cid_sel.append(cid_int)
    return pred_sel, gt_sel, cid_sel


def save_question_bundle(
    q_dir: str,
    q_idx: int,
    question: str,
    answer: str,
    focus_ids: List[int],
    image_bgr: np.ndarray,
    pred_masks_sel,
    gt_masks_sel,
    pred_cids_sel,
    gt_cids_sel,
    id2bgr: Dict[int, tuple],
    id2title: Dict[int, str],
    alpha: float = 0.45,
):
    os.makedirs(q_dir, exist_ok=True)
    H, W = image_bgr.shape[:2]

    # --- merged (pred/gt) ---
    pred_color, _, pred_per_class = _build_merged_color_mask(pred_masks_sel, pred_cids_sel, id2bgr, H, W)
    gt_color,   _, gt_per_class   = _build_merged_color_mask(gt_masks_sel,   gt_cids_sel,   id2bgr, H, W)

    pred_overlay = _alpha_overlay(image_bgr, pred_color, alpha=alpha)
    gt_overlay   = _alpha_overlay(image_bgr, gt_color,   alpha=alpha)

    pred_cls_mask = _build_merged_class_id_mask(pred_masks_sel, pred_cids_sel, H, W)
    gt_cls_mask   = _build_merged_class_id_mask(gt_masks_sel,   gt_cids_sel,   H, W)

    pred_overlay_pil = Image.fromarray(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    gt_overlay_pil   = Image.fromarray(cv2.cvtColor(gt_overlay,   cv2.COLOR_BGR2RGB))
    pred_labeled_pil = _draw_labels_from_mask(pred_overlay_pil, pred_cls_mask, id2title)
    gt_labeled_pil   = _draw_labels_from_mask(gt_overlay_pil,   gt_cls_mask,   id2title)
    pred_overlay_label = cv2.cvtColor(np.array(pred_labeled_pil), cv2.COLOR_RGB2BGR)
    gt_overlay_label   = cv2.cvtColor(np.array(gt_labeled_pil),   cv2.COLOR_RGB2BGR)
    # legend：题级只显示该题涉及的类别（pred_per_class/gt_per_class 的 key）
    pred_present_ids = list(pred_per_class.keys())
    gt_present_ids   = list(gt_per_class.keys())
    pred_overlay_label = _draw_legend(pred_overlay_label, pred_present_ids, id2title, id2bgr)
    gt_overlay_label   = _draw_legend(gt_overlay_label,   gt_present_ids,   id2title, id2bgr)

    cv2.imwrite(os.path.join(q_dir, "pred_merged_color.png"), pred_color)
    cv2.imwrite(os.path.join(q_dir, "pred_overlay.jpg"), pred_overlay)
    cv2.imwrite(os.path.join(q_dir, "pred_overlay_label.jpg"), pred_overlay_label)

    cv2.imwrite(os.path.join(q_dir, "gt_merged_color.png"), gt_color)
    cv2.imwrite(os.path.join(q_dir, "gt_overlay.jpg"), gt_overlay)
    cv2.imwrite(os.path.join(q_dir, "gt_overlay_label.jpg"), gt_overlay_label)

    # --- by_class ---
    by_class_pred = os.path.join(q_dir, "by_class", "pred")
    by_class_gt   = os.path.join(q_dir, "by_class", "gt")
    os.makedirs(by_class_pred, exist_ok=True)
    os.makedirs(by_class_gt, exist_ok=True)

    for cid, color_img in pred_per_class.items():
        title = id2title.get(cid, f"class_{cid}").replace("/", "_")
        name = f"{cid:03d}_{title}"
        cv2.imwrite(os.path.join(by_class_pred, f"{name}_color.png"), color_img)
        ov = _alpha_overlay(image_bgr, color_img, alpha=alpha)
        cv2.imwrite(os.path.join(by_class_pred, f"{name}_overlay.jpg"), ov)

    for cid, color_img in gt_per_class.items():
        title = id2title.get(cid, f"class_{cid}").replace("/", "_")
        name = f"{cid:03d}_{title}"
        cv2.imwrite(os.path.join(by_class_gt, f"{name}_color.png"), color_img)
        ov = _alpha_overlay(image_bgr, color_img, alpha=alpha)
        cv2.imwrite(os.path.join(by_class_gt, f"{name}_overlay.jpg"), ov)

    # --- meta.json for this question ---
    q_meta = {
        "q_index": int(q_idx),
        "question": question,
        "answer": answer,
        "focus_category_ids": [int(x) for x in focus_ids],
    }
    with open(os.path.join(q_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(q_meta, f, ensure_ascii=False, indent=2)

def _build_merged_color_mask(masks_list, category_ids, id2bgr, H, W):
    """
    masks_list: iterable of [H,W] 0/1 (torch/int)
    category_ids: iterable of class ids, len == len(masks_list)
    返回:
      merged_bgr: HxWx3 uint8
      present_ids: set
      per_class_bgr: {cid: HxWx3 uint8}  # 该类的合并彩色mask（同类多个实例合并）
    """
    merged = np.zeros((H, W, 3), dtype=np.uint8)
    present = set()
    per_class_bin = defaultdict(lambda: np.zeros((H, W), dtype=np.uint8))

    # 先把同类的二值mask合并
    for m, cid in zip(masks_list, category_ids):
        cid = int(cid)
        present.add(cid)
        if torch.is_tensor(m):
            m_np = m.detach().cpu().numpy()
        else:
            m_np = np.asarray(m)
        m_bin = (m_np > 0).astype(np.uint8)
        per_class_bin[cid] = np.maximum(per_class_bin[cid], m_bin)

    # 再生成 per-class 彩色图，并写入 merged（后写覆盖前写）
    per_class_bgr = {}
    for cid, binmask in per_class_bin.items():
        color = id2bgr.get(cid, (255, 255, 255))
        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        color_img[binmask == 1] = color
        per_class_bgr[cid] = color_img
        # 覆盖写到 merged（如果像素同时属于多个类，以后出现的类覆盖）
        merged[binmask == 1] = color
    return merged, present, per_class_bgr

def _alpha_overlay(image_bgr, color_mask_bgr, alpha=0.45):
    """
    image_bgr: HxWx3 uint8
    color_mask_bgr: HxWx3 uint8
    """
    if image_bgr is None:
        return None
    return cv2.addWeighted(image_bgr, 1 - alpha, color_mask_bgr, alpha, 0)

def _build_merged_class_id_mask(masks_list, category_ids, H, W):
    """
    把 instance masks 合并成单张 class-id mask:
      - 同类多个实例取 union
      - 输出 cls_mask: [H,W], uint16/uint8
      - 若像素被多个类覆盖：后写覆盖前写（和你 merged color 一致）
    """
    cls_mask = np.zeros((H, W), dtype=np.int32)

    # 先把同类合并成二值 union
    per_class_bin = defaultdict(lambda: np.zeros((H, W), dtype=np.uint8))
    for m, cid in zip(masks_list, category_ids):
        cid = int(cid)
        if torch.is_tensor(m):
            m_np = m.detach().cpu().numpy()
        else:
            m_np = np.asarray(m)
        m_bin = (m_np > 0).astype(np.uint8)
        per_class_bin[cid] = np.maximum(per_class_bin[cid], m_bin)

    # 再写回 cls_mask（后写覆盖前写）
    for cid, binmask in per_class_bin.items():
        cls_mask[binmask == 1] = cid

    return cls_mask

def _draw_labels_from_mask(overlay_img: Image.Image,
                          cls_mask: np.ndarray,
                          class_map: Dict[int, str]) -> Image.Image:
    img = overlay_img.copy()
    draw = ImageDraw.Draw(img)

    W, H = img.size
    base_font_size = max(16, int(min(W, H) / 32))

    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    for cid in np.unique(cls_mask):
        cid = int(cid)
        if cid == 0:
            continue

        ys, xs = np.where(cls_mask == cid)
        if xs.size == 0:
            continue

        x_min = xs.min(); x_max = xs.max()
        y_min = ys.min(); y_max = ys.max()

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        name = class_map.get(cid, f"class_{cid}")
        text = f"{cid}: {name}"

        tx = int(cx)
        ty = int(cy - base_font_size * 0.5)

        draw.text(
            (tx, ty),
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    return img

def _draw_legend(image_bgr, present_class_ids, id2title, id2bgr, max_items=30):
    """
    在左上角画图例：色块 + 'id: title'
    白字 + 黑描边（先画黑，再画白，保证白字不被盖住）
    """
    if image_bgr is None:
        return None
    out = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    x0, y0 = 10, 10
    box_w, box_h = 14, 14
    line_h = 20

    # 稳定排序一下（可选）
    ids = list(present_class_ids)
    try:
        ids = sorted(ids)
    except Exception:
        pass
    ids = ids[:max_items]

    for idx, cid in enumerate(ids):
        y = y0 + idx * line_h
        color = id2bgr.get(int(cid), (255, 255, 255))
        cv2.rectangle(out, (x0, y), (x0 + box_w, y + box_h), color, -1)

        text = f"{int(cid)}: {id2title.get(int(cid), str(cid))}"
        org = (x0 + box_w + 8, y + box_h - 2)

        # 先黑描边（粗）
        cv2.putText(out, text, org, font, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        # 再白字（细）
        cv2.putText(out, text, org, font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)

    return out



def token_f1(pred: str, gt: str):
    """
    SQuAD-style token F1
    """
    pred_toks = _tokenize(pred)
    gt_toks = _tokenize(gt)
    if len(pred_toks) == 0 and len(gt_toks) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_toks) == 0 or len(gt_toks) == 0:
        return 0.0, 0.0, 0.0

    common = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / max(1, len(pred_toks))
    recall = num_same / max(1, len(gt_toks))
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

def _tokenize(s: str):
    s = _normalize_text(s)
    if not s:
        return []
    return s.split()

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    # 去掉多余空白
    s = re.sub(r"\s+", " ", s)
    # 你也可以按需去标点：这里保守一点，只去掉常见符号
    s = re.sub(r"[^\w\s<>]", " ", s)  # Keep special tokens such as <SEG>.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bleu4(pred: str, gt: str, smooth=True):
    """
    简易 BLEU-4（单参考），带平滑（避免 0）
    返回 BLEU-4 in [0,1]
    """
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(gt)
    if len(pred_toks) == 0 and len(ref_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    # brevity penalty
    c = len(pred_toks)
    r = len(ref_toks)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r / max(1, c)))

    precisions = []
    for n in range(1, 5):
        pred_ng = _ngrams(pred_toks, n)
        ref_ng = _ngrams(ref_toks, n)
        if len(pred_ng) == 0:
            precisions.append(0.0)
            continue

        pred_cnt = Counter(pred_ng)
        ref_cnt = Counter(ref_ng)
        overlap = pred_cnt & ref_cnt
        num = sum(overlap.values())
        den = sum(pred_cnt.values())

        if smooth:
            # add-1 smoothing
            p = (num + 1.0) / (den + 1.0)
        else:
            p = 0.0 if den == 0 else (num / den)
        precisions.append(p)

    # geometric mean of p1..p4
    score = bp * math.exp(sum(math.log(p + 1e-12) for p in precisions) / 4.0)
    return float(score)

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens)-n+1)]


def extract_text_gt_pred(input_dict, output_dict):
    """
    尽可能从 input/output 中拿到 gt 文本和预测文本
    返回 (gt, pred) 或 (None, None)
    """
    # GT: 常见字段
    gt = None
    for k in ["text_answers", "answers", "gt_answer", "answer", "conversations"]:
        if k not in input_dict:
            continue
        v = input_dict[k]
        if isinstance(v, str):
            gt = v
            break
        if isinstance(v, (list, tuple)) and len(v) > 0:
            # text_answers: list[str] 或 list[list[str]]
            if isinstance(v[0], str):
                gt = v[0]
                break
            if isinstance(v[0], (list, tuple)) and len(v[0]) > 0 and isinstance(v[0][0], str):
                gt = v[0][0]
                break
        # conversations: 结构复杂，略做适配
        if k == "conversations":
            try:
                # 约定：index=1 是 assistant gt
                if isinstance(v, list) and len(v) >= 2 and isinstance(v[1], dict) and "value" in v[1]:
                    gt = v[1]["value"]
                    break
            except Exception:
                pass

    # Pred: 常见字段（取不到就 None）
    pred = None
    for k in ["pred_text", "text_output", "output_text", "prediction", "generated_text"]:
        if k in output_dict:
            pv = output_dict[k]
            if isinstance(pv, str):
                pred = pv
                break
            if isinstance(pv, (list, tuple)) and len(pv) > 0 and isinstance(pv[0], str):
                pred = pv[0]
                break

    # 有些实现把 pred 文本塞回 input_dict
    if pred is None:
        for k in ["prediction", "pred", "pred_answer"]:
            if k in input_dict:
                pv = input_dict[k]
                if isinstance(pv, str):
                    pred = pv
                    break

    if gt is None or pred is None:
        return None, None
    return gt, pred


def dice_fg(pred_mask, gt_mask, ignore_index=255):
    """
    pred_mask, gt_mask: torch.Tensor [H,W]，二值/0-1，或 {0,1,255}
    dice 只算前景1，忽略 255
    """
    valid = gt_mask != ignore_index
    if valid.sum() == 0:
        return None  # 无有效像素

    pred_fg = ((pred_mask == 1) & valid).int()
    gt_fg = ((gt_mask == 1) & valid).int()

    inter = (pred_fg & gt_fg).sum().float()
    denom = pred_fg.sum().float() + gt_fg.sum().float()

    # no-object → Dice=1
    if denom == 0:
        return torch.tensor(1.0, device=gt_mask.device)
    return (2.0 * inter) / (denom + 1e-5)


if __name__ == "__main__":
    main(sys.argv[1:])
