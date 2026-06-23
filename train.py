import argparse
import logging
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from model.VISA_multiseg import VrshqForCausalLM
from model.univi.config import ModelConfig, DataConfig
from model.llava import conversation as conversation_lib      # 为了测试暂时注释这三行代码
from dataset.multi_dataset import HybridDataset, ValDataset, collate_fn
from dataset.rvos_clip_eval_dataset import RVOSEvalDataset
from dataset.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda, intersectionAndUnionGPU
)

def parse_args(args):
    parser = argparse.ArgumentParser(description="VISA Model Training")
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
    parser.add_argument("--eval_only", action="store_true", default=False)
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

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    ##################################################### 日志
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0: # 0, 1, 2, 3, 4, 5, 6, 7
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        # 定义日志存放路径
        log_path = os.path.join(args.log_dir, "train.log")

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
    model = VrshqForCausalLM.from_pretrained(
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
    if not args.eval_only: # 默认进行训练, False
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

    # Re-enable the complete SIRA alignment modules after PEFT freezes the base model.
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
    train_dataset = HybridDataset(
        tokenizer,
        args.vision_tower,
        samples_per_epoch=(args.batch_size # batch_size: 2, 这里默认1
        * args.grad_accumulation_steps # grad_accumulation_steps: 10, 这里默认32, 梯度累计次数：多少次前向传播后，反向传播一次
        * args.steps_per_epoch # steps_per_epoch: 500, 这里默认1500
        * world_size), # GPU device count, 原文默认8
        precision=args.precision,
        image_size=args.image_size, # 1024
        num_classes_per_sample=args.num_classes_per_sample, # num_classes_per_sample: 3, 每张图像默认训练最多三个问题
        # new but old for VISA, 默认是False, 不排除验证集构建
        exclude_val=args.exclude_val,

        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory, # explanatory: 0.1
        # new but old for VISA
        univi_sample_frame_range=args.univi_sample_frame_range,
        balance_sample=args.balance_sample,
        rvos_seg_data=args.rvos_seg_data,
        rvos_sample_ratio=args.rvos_sample_ratio,
        rvos_num_frames_sample_range=args.rvos_num_frames_sample_range,
        rvos_sample_policy=args.rvos_sample_policy,
        univi_data_list = args.univi_data_list,
        univi_data_ratio = args.univi_data_ratio,
        univi_max_image_len = args.univi_max_image_len,
        # multi_reason_seg_data=args.multi_reason_seg_data, # 多类分割与拓展使用
        # num_classes_per_question=args.num_classes_per_question,
        # use_expand_question_list=args.use_expand_question_list,
    )
    if args.no_eval == False:
        val_out_dirname = args.val_dataset if '_split' not in args.val_dataset else args.val_dataset.split('_split', 1)[0]
        val_dataset = ValDataset(
            tokenizer                = tokenizer,
            vision_tower             = args.vision_tower,
            image_size               = args.image_size,
            val_dataset              = args.val_dataset,
            use_sira_tokens          = True,
            )

        logging.info(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        logging.info(f"Training with {len(train_dataset)} examples.")

    ##################################################### 模型加速, deepspeed配置
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    ##################################################### 加载模型存档点, resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        logging.info(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
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
    train_iter = iter(train_loader)
    best_score, cur_ciou, cur_dice = 0.0, 0.0, 0.0

    if args.eval_only:
        giou, ciou, dice = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou, dice = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou
            cur_dice = dice if is_best else cur_dice

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_epoch{}_giou{:.3f}_ciou{:.3f}_dice{:.3f}.pth".format(
                            epoch, best_score, cur_ciou, cur_dice
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)

    #     # 目前设为每一轮都保存验证结果？评价指标有些许改动。
    #     # if args.no_eval or is_best:
    #     if True:
    #         save_dir = os.path.join(args.log_dir, "ckpt_model")
    #         if args.local_rank == 0:
    #             torch.save(
    #                 {"epoch": epoch},
    #                 os.path.join(
    #                     args.log_dir,
    #                     "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
    #                         best_score, cur_ciou
    #                     ),
    #                 ),
    #             )
    #             #if os.path.exists(save_dir):
    #             #    shutil.rmtree(save_dir)
    #         torch.distributed.barrier()
    #         model_engine.save_checkpoint(save_dir)

    # if args.no_eval == False:
    #     validate(val_loader, model_engine, epoch, writer, args)

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    # classification_losses = AverageMeter("ClassificationLoss", ":.4f") # 多类时使用
    # variation_losses = AverageMeter("VariationLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            # classification_losses, # 多类时使用
            # variation_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()

    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):      # 梯度累积次数
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = [i.half() for i in input_dict["images"]]
                input_dict["images_clip"] = [i.half() for i in input_dict["images_clip"]]
            elif args.precision == "bf16":
                input_dict["images"] = [i.bfloat16() for i in input_dict["images"]]
                input_dict["images_clip"] = [i.bfloat16() for i in input_dict["images_clip"]]
            else:
                input_dict["images"] = [i.float() for i in input_dict["images"]]
                input_dict["images_clip"] = [i.float() for i in input_dict["images_clip"]]

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            # classification_loss = output_dict["classification_loss"] # 多类时使用
            # variation_loss = output_dict["variation_loss"]

            losses.update(loss.item(), input_dict["input_ids"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["input_ids"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["input_ids"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["input_ids"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["input_ids"].size(0))
            # classification_losses.update(
            #     classification_loss.item(), input_dict["images"].size(0)
            # )
            # if hasattr(variation_loss, "item"):
            #     loss_value = variation_loss.item()
            # else:
            #     loss_value = variation_loss
            # variation_losses.update(loss_value, input_dict["images"].size(0)) # 多类时使用

            model.backward(loss)
            model.step()

        # 测量经过时间, measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                # classification_losses.all_reduce() # 多类时使用
                # variation_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step + epoch * args.steps_per_epoch)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step + epoch * args.steps_per_epoch)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step + epoch * args.steps_per_epoch
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step + epoch * args.steps_per_epoch
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step + epoch * args.steps_per_epoch)
                # writer.add_scalar(
                #     "train/classification_loss", classification_losses.avg, global_step
                # )
                # writer.add_scalar(
                #     "train/variation_loss", variation_losses.avg, global_step
                # ) # 多类时使用
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step + epoch * args.steps_per_epoch
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step + epoch * args.steps_per_epoch
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            # classification_losses.reset() # 多类时使用
            # variation_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step + epoch * args.steps_per_epoch)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)

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

        # --------- prediction & gt ---------
        pred_masks = output_dict["pred_masks"]          # list, len=1
        if isinstance(pred_masks, list):
            pred_masks = pred_masks[0]
        if pred_masks.dim() == 4:   # [Ki,1,H,W]
            pred_masks = pred_masks[:,0]
        gt_masks = output_dict["gt_masks"][0].int()     # [N, H, W]
        pred_masks = (pred_masks > 0).int()             # [N, H, W]

        # print("\nimage_paths:", input_dict["image_paths"], "  pred:", tuple(pred_masks.shape), pred_masks.dtype, pred_masks.device, "  gt  :", tuple(gt_masks.shape), gt_masks.dtype, gt_masks.device)

        assert pred_masks.shape == gt_masks.shape

        if pred_masks.numel() == 0:
            continue

        # --------- IoU & Dice ---------
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        dice_sum, dice_cnt = 0.0, 0

        for pred_i, gt_i in zip(pred_masks, gt_masks):
            # IoU（你原来的逻辑）
            inter_i, union_i, _ = intersectionAndUnionGPU(
                pred_i.contiguous(), gt_i.contiguous(), 2, ignore_index=255
            )
            intersection += inter_i
            union += union_i
            acc_iou += inter_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

            # ---------- Dice（前景类 1）----------
            valid = gt_i != 255
            if valid.sum() == 0:
                continue

            gt_fg = ((gt_i == 1) & valid).int()
            pred_fg = ((pred_i == 1) & valid).int()

            inter = (gt_fg & pred_fg).sum().float()
            denom = gt_fg.sum().float() + pred_fg.sum().float()

            # no-object → Dice=1
            if denom == 0:
                dice_i = torch.tensor(1.0, device=gt_i.device)
            else:
                dice_i = (2.0 * inter) / (denom + 1e-5)

            dice_sum += dice_i
            dice_cnt += 1

        # --------- batch 汇总 ---------
        intersection = intersection.cpu().numpy()
        union = union.cpu().numpy()
        acc_iou = (acc_iou / gt_masks.shape[0]).cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=gt_masks.shape[0])

        if dice_cnt > 0:
            dice_meter.update((dice_sum / dice_cnt).item(), n=dice_cnt)

    # --------- distributed reduce ---------
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    dice_meter.all_reduce()

    # --------- final metrics ---------
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] # 全局 IoU，且原本包含两个类别，只取前景类1
    giou = acc_iou_meter.avg[1] # 样本级 IoU 的平均值，且原本包含两个类别，只取前景类1
    dice = dice_meter.avg # 样本级 dice 的平均值，只包含前景类的标量

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/dice", dice, epoch)
        logging.info(
            "giou: {:.4f}, ciou: {:.4f}, dice: {:.4f}".format(giou, ciou, dice)
        )

    return giou, ciou, dice

def rvos_validate(val_loader, model_engine, epoch, writer, args):
    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
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
        pred_masks = output_dict["pred_masks"]
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1
        for mask_i, output_path_i in zip(output_list, input_dict['image_paths']):
            assert output_path_i.endswith('.png'), f'output_path_i: {output_path_i} must end with .png'
            # output_path_i = "/" + os.path.join(*output_path_i.split('/')[:-3], str(epoch), *output_path_i.split('/')[-3:])
            # save mask_i to output_path_i
            mask_i = mask_i.cpu().numpy().astype(np.float32)
            mask_i = Image.fromarray(mask_i * 255).convert('L')
            os.makedirs(os.path.dirname(output_path_i), exist_ok=True)
            mask_i.save(output_path_i)


if __name__ == "__main__":
    main(sys.argv[1:])
