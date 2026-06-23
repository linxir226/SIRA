import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools import mask

from model.llava import conversation as conversation_lib

from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (REASON_ANSWER_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
                    EXPLANATORY_QUESTION_LIST, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, convert2imagesplit)


class SurgRSDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="SurgRS|surgrs_train.json",
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample


        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # 问题需要使用模板，答案目前需要特制的
        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST
        self.long_question_list = UNIFIED_LONG_QUESTION_LIST
        self.answer_list = REASON_ANSWER_LIST

        reason_seg_data, json_name = reason_seg_data.split("|")
        self.base_image_dir = os.path.join(base_image_dir, reason_seg_data)
        json_file_name = os.path.join(
            self.base_image_dir, json_name
        )
        with open(json_file_name, "r") as f:
            reason_file = json.load(f)
        self.reason_part_seg_data = reason_file["data"]
        print("# of part_reason_seg samples: ", len(reason_file["data"]))

    def __len__(self):
        return self.samples_per_epoch

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

    def __getitem__(self, idx):
        # 完全随机采样
        idx = random.randint(0, len(self.reason_part_seg_data) - 1)
        image_info = self.reason_part_seg_data[idx]

        image_path = image_info["image_path"] # "valid/video_41/rgb/000000000.png"

        question = image_info["questions"] # ["Which left-sided device component is providing the primary lateral traction on tissue?", ... ]
        gt_answer = image_info["answers"] # [seg]对应的掩码 len=[seg].num
        text_answers = image_info["text_answers"] # ["Left tool clasper [SEG], Left tool wrist [SEG]. These left-side components are selected because the distal jaw (Left tool clasper) is clearly engaging tissue for lateral traction while the articulating joint (Left tool wrist) adjusts the angle to hold the exposure.", ... ]

        image_path = os.path.join(self.base_image_dir, image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 随机抽取图像并按照clip的方式预处理 preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # 图像应用sam格式处理 preprocess image for sam 一个几何变换，把图像最长边缩放到 image_size，同时保持比例。
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        masks = []

        # 每一项任务挑选有限数量的问题 NOTE: maximum num_classes_per_sample questions per sample
        if len(question) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(question))),
                size=self.num_classes_per_sample,
                replace=False,
            ) # 例子：sampled_inds = np.array([0, 2, 4])
        else:
            sampled_inds = list(range(len(question)))

        sampled_sents = np.vectorize(question.__getitem__)(sampled_inds).tolist() # 例子：sampled_sents = [question[0], question[2], question[4]]

        sampled_answers = gt_answer
        sampled_masks = masks # 这里 sampled_masks 和 masks 是同一个 list 对象, 对masks处理等同于对sampled_masks处理
        sampled_text_answers = text_answers # 暂未采样，说明没有超过三个的情况，全部走else

        questions = []
        answers = []
        category_ids = []
        # 生成问题/答案文本,简单加上分割要求，并未形成人机对话 + 解析 RLE mask
        if len(question) != 0:
            for text, answer_list, text_answer in zip(
                sampled_sents, sampled_answers, sampled_text_answers
            ):
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))

                query_tokens = "<QUERY>" * 64
                _text_answer = text_answer.replace("[SEG]", "<SEG>") + query_tokens
                answers.append(_text_answer)

                for answer in answer_list:
                    rle = answer["segmentation"]
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    masks.append(m)

                    category_ids.append(answer["category_id"])

        # 构造 conversation prompt 提示列表
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # 未使用

        # 加载多种对话模板
        i = 0
        while i < len(questions):
            questions[i] = convert2imagesplit(questions[i], 1) # new, 把可能出现的<video>替换为(0)<image>, 因为这里实际是图像处理, 而且只有一张图
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # 预处理后图像 (标准化 + pad 到 1024×1024)
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)         # 3, h, w
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        category_ids = torch.tensor(category_ids)

        num_image_token = 1
        return (
            image_path,
            image, # tensor([3, 1024, 1024]) SAM规范预处理后图像 (标准化 + pad 到 1024×1024)
            torch.stack([image_clip] * num_image_token, dim=0), # tensor([3, 224, 224]) CLIP 格式预处理图像 (224×224 或 336×336) 用于LLM
            conversations, # list of str, ["SYSTEM: abc <human>: question <gpt>: answer",] 完整对话
            masks, # tensor([[seg].num, H, W]) 分割掩码
            label, # tensor([H, W]) 标签图 (初始全为 ignore=255)
            resize, # (H, W) 最长边规范调整后图像尺寸
            questions, # list of str 实际生成的问题文本（填入初步模板后的版本）
            sampled_sents, # list of str 原始问题文本
            category_ids, # tensor([1,1,...,num_classes_per_sample]) 每个[seg]类别ID
            # [0],
        )

# conversations示例:
# [SYSTEM: You are a helpful multimodal assistant. <SEP>
# USER: (0)<image>
# Where is the lesion region?<SEP>
# ASSISTANT: The lesion is indicated by <SEG>.<SEP>,]
