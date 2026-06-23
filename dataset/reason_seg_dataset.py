import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib

from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
                    EXPLANATORY_QUESTION_LIST, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, convert2imagesplit)


class ReasonSegDataset(torch.utils.data.Dataset):
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
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST
        self.long_question_list = UNIFIED_LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:       # 加载问题和答案
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

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
        # 不使用当前idx，随机选取图片路径
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # 随机抽取图像并按照clip的方式预处理 preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # 对于每轮每张图片，随机采样不超过3个问题（refs） and get masks
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            ) # 例子：sampled_inds = np.array([0, 2, 4])
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist() # 例子：sampled_sents = [question[0], question[2], question[4]]
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ] # 这里目标都相同，复制3份

        # 图像应用sam格式处理 preprocess image for sam 一个几何变换，把图像最长边缩放到 image_size，同时保持比例。
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # 如果本次随机图像含有解释, self.explanatory即0.1概率选择2, 否则随机选择0或1, 对本轮所有问题有效
        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:     # 对于选中的每一个问题，将其加载到模板中
            if is_sentence: # 推理分割中都是句子
                question_template = random.choice(self.long_question_list)     # 随机选择一个问题模板
                questions.append(question_template.format(sent=text))        # 将reference引入问题模板的<sent>位置
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(sent=text.lower()))

            # 有三种回答方式，第一种0是直接分割目标，第二种1是分割目标+解释说明，第三种2是解释说明而不进行分割, add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]         # 正确答案
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = random.choice(self.explanatory_question_list).format(sent=text.lower()) # 修改问题为分割+解释
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_VIDEO_TOKEN + "\n" + text # 修改问题为纯解释
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

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

        image_name = image_path.split("/")[-1]        # masks and labels
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)         # 3, h, w
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        if self.explanatory != -1 and img_name in self.img_to_explanation and choice == 2:
            num_image_token = 1
        else:
            num_image_token = 2

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
            # [0],
        )

# conversations示例:
# [SYSTEM: You are a helpful multimodal assistant. <SEP>
# USER: (0)<image>
# Where is the lesion region?<SEP>
# ASSISTANT: The lesion is indicated by [SEG].<SEP>,]