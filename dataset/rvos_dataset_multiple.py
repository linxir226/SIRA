###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import itertools
import json
import os
import os.path as osp
import pickle
import sys
import cv2
import time
import random
import torch
import math
import torch.nn.functional as F
from pprint import pprint
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import pycocotools.mask as maskUtils

from model.llava import conversation as conversation_lib

from model.segment_anything.utils.transforms import ResizeLongestSide
from transformers import CLIPImageProcessor
from PIL import Image

from .d2_datasets.refytvos_utils import load_refytvos_json
from .d2_datasets.mevis_utils import load_mevis_json
from .utils import (
    DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN,
    UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, ANSWER_LIST,
    convert2imagesplit
)

from .dataset_config import RVOS_DATA_INFO as _DATA_INFO
from .dataset_config import RVOS_ROOT

def get_zero_image(processor): # 未使用
    i = Image.new('RGB', (224, 224), (0, 0, 0))
    return processor.preprocess(i, return_tensors='pt')['pixel_values'][0]


class RVOSDataset(torch.utils.data.Dataset):
    # davis17_train, refytvos_train, mevis_train
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self,
        tokenizer,
        vision_tower,
        samples_per_epoch       : int   = 500 * 8 * 2 * 10,
        precision               : str   = "fp32",
        image_size              : int   = 224,
        num_classes_per_sample  : int   = 3,
        num_frames_sample_range : int   = "8,12", # new, 一个字符串 "8,12" 表示每次采样的帧数范围 [8, 12]
        rvos_sample_policy      : str   = "uniform", # new, 从不同子数据集采样的策略，"uniform" 或 "random"
        rvos_seg_data           : str   = "mevis_train||refytvos_train||davis17_train||revos_train",
        rvos_sample_ratio       : str   = '4000||15000||400||6000',
        rvos_sample_list        : list  = [],
        prob_no_in_video        : float = 0.05, # new, 有一定概率采样“video 里没有这个目标”的负例表达
    ):
        self.root = RVOS_ROOT
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.samples_per_epoch = samples_per_epoch
        self.transform = ResizeLongestSide(image_size) # 一个几何变换，把图像最长边缩放到 image_size，同时保持比例。
        self.num_frames_sample_range = [int(x) for x in num_frames_sample_range.split(",")] # "8,12" → [8, 12], 表示每次采样的帧数范围 [8, 12]
        assert len(self.num_frames_sample_range) == 2 and self.num_frames_sample_range[0] <= self.num_frames_sample_range[1], f"invalid num_frames_sample_range {num_frames_sample_range}"
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) # CLIP 图像预处理器
        self.prob_no_in_video = prob_no_in_video

        assert rvos_sample_policy in ["random", "uniform"], f"invalid rvos_sample_policy {rvos_sample_policy}"
        self.rvos_sample_policy = rvos_sample_policy
        self.rvos_sample_list = rvos_sample_list
        self.num_be_called = 0 # 记录 __getitem__ 被调用次数，用于按 epoch 控制之类。

        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST
        self.long_question_list = UNIFIED_LONG_QUESTION_LIST

        self.answer_list = ANSWER_LIST

        self.rvos_seg_ds_list = rvos_seg_data.split("||")
        rvos_sample_ratio = np.array([float(x) for x in rvos_sample_ratio.split("||")])
        self.rvos_sample_ratio = rvos_sample_ratio / rvos_sample_ratio.sum()     # 每个子数据集被采样到的概率
        self.rvos_seg_data = {}
        # 逐个子数据集加载元信息到内存
        for dataset in self.rvos_seg_ds_list:
            assert dataset in _DATA_INFO.keys(), f"dataset {dataset} not found!"
            print(f"loading dataset {dataset} into memory...")
            image_root, json_file = _DATA_INFO[dataset]
            image_root = osp.join(self.root, image_root)
            json_file = osp.join(self.root, json_file)       # 处理相同格式的数据集
            if 'mevis' in dataset or 'revos' in dataset or 'lvvis' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_mevis_json(image_root, json_file, dataset, is_train = True)
            elif 'refytvos' in dataset or 'davis' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_refytvos_json(image_root, json_file, dataset)
            else:
                raise ValueError(f"Unknown dataset name: {dataset}")
            assert is_train, 'only support training mode for now'
            print(f'Loaded {dataset} dataset, with {len(metas)} expressions, {len(vid2metaid)} videos')

            self.rvos_seg_data[dataset] = {
                'image_root': image_root, # 该数据集训练图像根目录
                'json_file' : json_file, # 该数据集训练 json 文件路径
                'metas'     : metas, # [{}...]视频所有元, 详见load_mevis_json函数
                'mask_dict' : mask_dict, # 掩码表, 与anno_id对应, {"1000000"即anno_id[i]:[{"size":[360, 640], "counts":"CUL"}, ...], ...}
                'is_train'  : is_train, # True
                'vid2metaid': vid2metaid, # {'377b1c5f365c': [0, 1, 2, ...], ...}, video_id -> [meta_id, ...], 长度为该视频的问题数量
            }

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        data = self.sample_data() # 帧采样

        # 从路径读取图像路径并clip编码用于LLM, frames that are sent to the clip encoder and sam
        frame_list = [cv2.imread(x) for x in data['video_frame_for_llm']]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        frame_clip_list = [self.clip_image_processor(x, return_tensors="pt")["pixel_values"][0] for x in frame_list]     # list: T个, 3, h, w
        video_len = len(frame_clip_list)

        # target frame sent to sam encoder
        # seg_frame = cv2.imread(data['seg_frame_path'])
        # seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
        # seg_frame_clip_ = self.clip_image_processor(seg_frame, return_tensors="pt")["pixel_values"][0]

        # 从路径读取图像路径并sam编码, 实际上两者的采样帧是完全一致的。multiple sample frame sent to the sam encoder
        frame_list = [cv2.imread(x) for x in data['video_frame_path_list']]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        # seg_frame_clip = [self.clip_image_processor(x, return_tensors="pt")["pixel_values"][0] for x in frame_list]
        seg_frame_sam = [self.transform.apply_image(x) for x in frame_list]
        resize = seg_frame_sam[0].shape[:2]
        frame_sam_list = [self.preprocess(torch.from_numpy(x).permute(2, 0, 1)).contiguous() for x in seg_frame_sam]     # list: T个, 3, h, w

        # conditional_frame_ids = [data['video_frame_for_llm'].index(x) for x in data['video_frame_path_list']]    # 每个条件帧的索引
        # conditional_frame_ids = [data['video_frame_path_list'].index(data['seg_frame_path'])]
        conditional_frame_ids = [0]

        questions = []
        answers = []
        conversations = []
        masks = []
        conv = conversation_lib.default_conversation.copy()
        for exp, mask in data['exp_mask_pairs']:
            text = exp.strip()
            assert len(text.split('||')) == 1
            if text[-1] == "?": # 都是句子，短句子会补一个问号
                question = random.choice(self.long_question_list).format(sent=text)
            else:
                question = random.choice(self.short_question_list).format(sent=text)
            question = convert2imagesplit(question, video_len) # 将视频标记token转换为采样帧长度连续图片标记token, "(0)<image>, (1)<image>, ..."
            # question = convert2imagesplit(question, 1)

            seg_replace = ", ".join(f'({i}) [SEG]' for i in range(video_len))
            # answer = random.choice(self.answer_list).format(seg=seg_replace)
            answer = random.choice(self.answer_list).replace("[SEG]", seg_replace) # 回答模板是固定的, 和标记token一样将[SEG]替换为采样帧长度个[SEG], "(0) [SEG], (1) [SEG], ..."

            questions.append(question)
            answers.append(answer)

            conv.messages = []
            conv.append_message(conv.roles[0], questions[-1])
            conv.append_message(conv.roles[1], answers[-1])
            conversations.append(conv.get_prompt())

            masks.append(mask)

        masks = torch.from_numpy(np.stack(masks, axis=0))     # (num_classes, num_frame, H, W)
        label = torch.ones(masks.shape[-2], masks.shape[-1]) * self.ignore_label

        return (
            ','.join(data['video_frame_path_list']),
            torch.stack(frame_sam_list, dim=0),          # tensor([T, 3, 1024, 1024]) SAM规范预处理后图像 (标准化 + pad 到 1024×1024)
            torch.stack(frame_clip_list, dim=0),         # tensor([T, 3, 224, 224]) CLIP 格式预处理图像 (224×224 或 336×336) 用于LLM # 不需要关键帧
            conversations,                               # list of str, ["SYSTEM: abc <human>: question <gpt>: answer",] 完整对话, 多帧对应
            masks,                                       # tensor([exp.num, T, H, W]) 分割掩码, 单问题多个目标已合并
            label,                                       # tensor([H, W]) 全图背景标签
            resize,                                      # (H, W) 最长边规范调整后图像尺寸
            questions,                                   # list of str 实际生成的问题文本（填入初步模板后的版本）
            [exp for exp, _ in data['exp_mask_pairs']],  # list of str 原始问题文本
            # conditional_frame_ids,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def sample_cond_frame(self, sample_list, max_dist, num_frame): # 未使用
        target = np.random.choice(sample_list, 1, replace=False)[0]
        index = sample_list.index(target)

        if index - max_dist < 0:
            target_list = sample_list[0: 2 * max_dist]
        elif index + max_dist >= len(sample_list):
            target_list = sample_list[-2 * max_dist:]
        else:
            target_list = sample_list[index - max_dist: index + max_dist]

        frame_ids_sam = np.random.choice(target_list, num_frame, replace=False).tolist()
        frame_ids_sam = sorted(frame_ids_sam)
        seg_frame_ids = np.random.choice(frame_ids_sam, 1, replace=False)[0]

        return frame_ids_sam, seg_frame_ids

    def sample_data(self,):
        ds         = np.random.choice(list(range(len(self.rvos_seg_ds_list))), p=self.rvos_sample_ratio) # 按照既定概率选择一个子数据集
        ds         = self.rvos_seg_ds_list[ds]
        metas      = self.rvos_seg_data[ds]['metas']         # 数据集中的reference以及各帧信息
        mask_dict  = self.rvos_seg_data[ds]['mask_dict']     # 数据集中的mask信息
        image_root = self.rvos_seg_data[ds]['image_root']    # 数据集图像根目录
        vid2metaid = self.rvos_seg_data[ds]['vid2metaid']    # 视频索引：expression索引

        # 采样一个视频 sample a video
        vid = np.random.choice(list(vid2metaid.keys()))
        meta_ids = vid2metaid[vid]      # 视频的问题（表达式）所有ids, the expression id in this video
        # 随机选择不超过3条reference, random choose self.num_classes_per_sample indices
        meta_ids = np.random.choice(meta_ids, min(self.num_classes_per_sample, len(meta_ids)), replace=False)
        video_name = metas[meta_ids[0]]['video']        # 读取对应的视频名称(第一个结果)
        assert all([metas[meta_id]['video'] == video_name for meta_id in meta_ids]), "video name not match" # 确保所有选择的reference属于同一个视频

        record = {}
        vid_dict_first = metas[meta_ids[0]] # 读取对应的视频元信息(第一个结果)
        record["file_names"] = [       # 加载视频中的所有帧路径
            os.path.join(image_root, 'JPEGImages', vid_dict_first['video'], vid_dict_first["frames"][i]+ '.jpg')
            for i in range(vid_dict_first["length"])
        ]
        record["length"] = vid_dict_first["length"] # 视频长度
        # 随机选择 self.num_frames_per_sample 帧
        # self.num_frames_sample_range
        if len(self.rvos_sample_list) > 0:       # 确定每个视频样本抽取的帧数
            num_frames_per_sample = self.rvos_sample_list[self.num_be_called % len(self.rvos_sample_list)]
            self.num_be_called += 1
        else:      # 否则随机从8-12帧中抽取
            num_frames_per_sample = np.random.randint(self.num_frames_sample_range[0], self.num_frames_sample_range[1] + 1)

        # 具体的采样策略
        if vid_dict_first["length"] > num_frames_per_sample:
            if self.rvos_sample_policy == "random":
                frame_ids = np.random.choice(vid_dict_first["length"], num_frames_per_sample, replace=False).tolist()
                frame_ids = sorted(frame_ids) # 随机采样num_frames_per_sample帧并排序
            elif self.rvos_sample_policy == "uniform":
                num_length = vid_dict_first["length"]
                split_point = np.linspace(0, num_length, num=num_frames_per_sample + 1, dtype=int)      # 从0开始均匀采样num_frames_per_sample + 1帧
                frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(num_frames_per_sample)]      # 每两个数字之间随机采样一帧，从而得到均匀分布的采样帧
        else:
            frame_ids = list(range(vid_dict_first["length"])) # 所有帧

        # num_frames_llm = num_frames_per_sample
        # frame_ids_llm = np.random.choice(frame_ids[1: ], num_frames_llm - 1, replace=False).tolist()
        # frame_ids_llm = [frame_ids[0]] + sorted(frame_ids_llm)

        # 将取出的所有视频帧送入chat-univi中进行处理
        video_frame_for_llm = [record['file_names'][i] for i in frame_ids]

        # 输入SAM2采用和输入LLM相同的视频帧
        video_frame_path_list = [record['file_names'][i] for i in frame_ids]
        image_shape = cv2.imread(record["file_names"][0]).shape[:2]

        # 提取不同exp 的 mask
        exp_mask_pairs = []
        for meta_id in meta_ids:      # 对于每一条问题表达exp
            vid_dict = metas[meta_id]
            assert vid_dict['video'] == video_name, "video name not match"
            assert vid_dict['length'] == vid_dict_first['length'], "video length not match"
            anno_ids = vid_dict['anno_id'] # 当前问题表达指代物体掩码编码, 对于该数据集唯一 [1000000, 1000001, 1000001, 1000002, ...]
            obj_ids = vid_dict['obj_id'] # 当前问题表达指代物体编号 [0, 1, 1, 2, ...]
            exp = vid_dict['exp'] # 当前问题表达 4 lizards moving around
            if 'lvvis' in ds:
                exp = exp.replace('_', ' ')

            m_final_list = []
            for seg_frame_id in frame_ids:
                m_final = np.zeros(image_shape, dtype=np.uint8)
                for x, obj_id in zip(anno_ids, obj_ids):      # 对于每个物体的标注索引和类别索引
                    segm = mask_dict[x][seg_frame_id]    # x表示物体掩码编码索引，seg_frame_id表示帧索引，对应帧的mask标注
                    if segm is not None:
                        m = maskUtils.decode(segm) # [H, W] 或 [H, W, 1]
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        m_final = m_final | m     # 对应某一帧的mask标注
                m_final_list.append(m_final)

            m_final_list = np.stack(m_final_list, axis=0)  # (sample_num_frame, H, W)
            exp_mask_pairs.append((exp, m_final_list))

        data = {
            "video_name"           : video_name,
            "video_frame_path_list": video_frame_path_list,    # video frames for sam
            "video_frame_for_llm"  : video_frame_for_llm,     # video frames for llm
            # "seg_frame_path"       : seg_frame_path,
            "exp_mask_pairs"       : exp_mask_pairs, # [(exp_0, masks_0), (exp_1, masks_1), ...], masks_n = (T, H, W), 已将该目标所有掩码细分目标考虑在内
        }

        return data

