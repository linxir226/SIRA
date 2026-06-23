import glob
import json
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
# from model.videollama2 import conversation as conversation_lib

# from model.videollama2.mm_utils import tokenizer_multimodal_token as tokenizer_image_token
# from model.videollama2.mm_utils import tokenizer_image_token

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .surgrs_dataset import SurgRSDataset
from .utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    convert2imagesplit, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, ANSWER_LIST
)
from .vqa_dataset import VQADataset
from .chatunivi_dataset import ChatUniviDataset
# from .rvos_dataset import RVOSDataset
from .rvos_dataset_multiple import RVOSDataset
from .random_list import get_random_list
from .dataset_config import LISA_ROOT


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    category_ids_list = [] ### 新增：存储每个对话的类别ID
    offset_list = [0] # 用于记录每个对话的偏移量
    cnt = 0 # 记录对话的总数
    inferences = []
    num_frame_list = [] # new, 用于记录每个对话的帧数
    num_conv_list = [] # new, 用于记录每个图像的对话轮数
    # conditional_frame_ids = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        category_ids,
        # condition_frame_ids,
        inference,
    ) in batch:          # 对于一个batch中的图像的集中处理
        image_path_list.append(image_path)

        if images.ndim > 3:
            pass           # T, 3, 1024, 1024 (视频)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        assert images.ndim == 4
        images_list.append(images)    # T, 3, 1024, 1024  # 构建成列表

        if images_clip.ndim == 3:
            images_clip = images_clip.unsqueeze(0)
        assert images_clip.ndim == 4
        images_clip_list.append(images_clip)    # 构建成列表
        num_frame = images_clip.shape[0]        # T, 3, 224, 224
        num_frame_list.append(num_frame) # [T, ...]

        conversation_list.extend(conversations) # 注意只有ta是extend
        label_list.append(label)          # 720, 1280
        num_conv_list.append(len(conversations)) # 由于图像调整成列表无法获知每个图像对话轮数，故在此记录下来 [num_conv, ...]

        if masks.ndim == 3:  # [num_classes, H, W]
            if masks.shape[0] == 0:  # [0, H, W] -> [num_classes, 0, H, W]
                masks = torch.stack([masks, ] * len(conversations), dim=0).float()
            else: # [num_classes, H, W] -> [num_classes, 1, H, W]
                masks = masks.unsqueeze(1).float()     # num_classes, 1, H, W
        assert masks.ndim == 4        # num_classes, num_frames, H, W 对于图像, num_classes对于图片是[seg]数量(多对话有序排列), 对于视频是该采样对话数量
        masks_list.append(masks.float()) # mask添加一维

        resize_list.append(resize)          # 14, 1024
        questions_list.append(questions)
        category_ids_list.append(category_ids)
        sampled_classes_list.append(sampled_classes) # 原始问题文本, 由于编写之初是类分割所以是这个名字

        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        # conditional_frame_ids.append(condition_frame_ids)

    # 如果use_mm_start_end是True，说明要在<image>前后加上开始（IM_START）和结束（IM_END）标志。遍历每一条对话，把原本的<image>替换成加了头尾的版本。
    if use_mm_start_end:      # <image> token -> <im_start> <image> <im_end> token
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    # 防止意外的含有<video> token未处理情况, 但舍弃编号, 尽量避免或做出修改
    for i in range(len(conversation_list)):
        if DEFAULT_VIDEO_TOKEN in conversation_list[i]:   # 若存在 <video> token 在 conversation 中
            if conversation_list[i].count(DEFAULT_VIDEO_TOKEN) == 1:      # 将<video>token替换为多个<image>token
                replace_video_token = DEFAULT_IMAGE_TOKEN * num_frame # 仅是最后一次，可能会出现错误
                conversation_list[i] = conversation_list[i].replace(DEFAULT_VIDEO_TOKEN, replace_video_token)
            else:
                raise ValueError("num video token > 1: ", conversation_list[i].count(DEFAULT_VIDEO_TOKEN))

    # 生成并将所有input_ids按照最长的补齐（pad），使它们shape一致，方便打包成一个batch。
    # input_ids是对话生成的嵌入索引
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(          # num_conversation, num_tokens
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # 生成attention_mask，pad的地方是0，真实的token是1。
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    # targets是复制一份input_ids嵌入索引，后面会把其中不需要计算loss的位置设置成IGNORE_INDEX。
    targets = input_ids.clone()

    # 根据对话类型选择不同的分隔符。llava_v1和llava_llama_2的分隔规则略有不同。
    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "    # 找到分割user和assistant的标记
    else:
        sep = "[/INST] "

    # 遍历每一条对话和对应的target。
    for conversation, target in zip(conversation_list, targets):
        # 实际长度
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 按照分隔符sep2（比如 "</s>"）切分对话轮次（每一轮问答）。
        rounds = conversation.split(conv.sep2)    # 对话轮数
        # cur_len初始是1（通常是因为第一个token是<s>或者特殊token）。
        # 开头的token标记为IGNORE_INDEX（不会算loss）。
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        # 遍历每一轮对话，如果是空（比如最后多了一个分隔符）就跳过。
        for i, rou in enumerate(rounds):     # 处理第一轮对话
            if rou == "":
                break
            # 每一轮应该能分成两个部分：一个是指令（instruction），一个是回答（answer）。
            # 检查确保能正确分成两部分，否则报错。
            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            # 把instruction部分补回分隔符，方便后面算长度。
            parts[0] += sep

            # 如果对话里包含图像标记，使用tokenizer_image_token来分词。
            # 计算这一轮总长度round_len，和仅指令部分长度instruction_len。
            # 注意这里-2是因为去掉特殊的开头结尾token（比如<s>、</s>）。
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))       # 对话的问题+答案部分的总体长度
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2   # 对话的问题部分长度
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX    # 将对话的问题部分的token屏蔽

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(cur_len, total_len)
            assert cur_len == total_len
        # delete 如果输入太长了，直接截断到允许的最大长度。

    return {
        # 原始图片路径列表，["../datasets/PathMR/images/case001.png",]。
        "image_paths": image_path_list,
        # SAM规范预处理后图像列表，[tensor([T, 3, 1024, 1024]), ...]。
        "images": images_list, #BS : T(or 1 for chatunivi) * 3 * H * W
        # CLIP 格式预处理图像列表，用于LLM，[tensor([T, 3, 224, 224]), ...]。
        "images_clip": images_clip_list, #BS : T * 3 * H * W
        # 嵌入的对话文本的token索引序列tensor([conv.num, L])，单张图像的每个对话嵌入算一个，长度长于batch。
        "input_ids": input_ids,      # num_conversation, ntokens
        # 嵌入的对话文本的token序列tensor([conv.num, L])，只不过指令部分被标记成IGNORE_INDEX，用于计算回答的loss。
        "labels": targets,
        # 指示input_ids中哪些位置是有效token（1），哪些是padding（0），tensor([conv.num, L])，用于Transformer结构中屏蔽无关计算。
        "attention_masks": attention_masks,
        # 分割掩码列表，[tensor([conv.num, T, H, W]),]。对于图像, conv.num对于图片是[seg]数量(多对话有序排列), 对于视频是该采样对话数量(因为当前每个对话只有一个)。
        "masks_list": masks_list, # [Conv*Frame*H*W, ...]
        # 标签图即原图列表 (初始全为 ignore=255)，[tensor([H, W]), ...]。长度为bs
        "label_list": label_list, # [H*W, ...]    gt label
        # 统一长边预处理后真实图像的形状大小列表，每个值为（960，1024）这样，[(H, W), ...]。 长度为bs
        "resize_list": resize_list,       # [14, 1024]
        # 记录input_ids对话中应该更换图像的位置，形如tensor([0, 2, 3, 6, ...])具体值，第i张图片的对话从offset[i]到offset[i+1]，长度len(input_ids)+1。【当前设定为最大3】
        "offset": torch.LongTensor(offset_list), #[0, num_conv0, num_conv1, ...]   例如：[0, 1, 3]中，0-1表示第0段对话属于第一个视频，1-3表示第1、2段对话属于第二个视频
        # 实际生成的问题文本列表，[[str, str, ...], ...]。
        "questions_list": questions_list,        # bs
        # 原始问题文本，[[str, str, ...], ...]。
        "sampled_classes_list": sampled_classes_list,
        # 当前模式是否是推理模式，True表示推理，False表示训练。
        "inference": inferences[0],
        # 完整对话列表，当前未添加图片前后修饰，[["SYSTEM: abc <human>: question <gpt>: answer",],]。
        "conversation_list": conversation_list,
        # 用于记录每个对话的帧数，[T, T, ...]。(bs,以图像或视频片段为单位)
        "num_frame_list": num_frame_list,      # bs
        # 用于记录每个图像的对话轮数，[conv.num, conv.num, ...]。(bs,以图像或视频片段为单位，用于核对)
        "num_conv_list": num_conv_list,      # bs
        # "cond_frame_list": conditional_frame_ids,    # None, [0], [0, 3, 7, ...]
        ### 新增：存储每个[seg]的类别ID，[tensor([1,1,...,num_classes_per_sample]),]
        "category_ids": category_ids_list,
    }


class  HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,        # 每个epoch中按一定比例采样不同的数据来进行预训练，并非直接使用所有的数据
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        rvos_seg_data="mevis_train||refytvos_train||davis17_train",
        rvos_sample_ratio='4000||15000||400',
        rvos_num_frames_sample_range="6,12",
        rvos_sample_policy="uniform",
        univi_data_list = "mimic||sqa||video",
        univi_data_ratio = "1||1||1",
        univi_max_image_len = 64,
        explanatory=0.1,
        univi_sample_frame_range="10,12",
        balance_sample=True,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()  # [9，3，3，1] / 16  分别针对4个数据集的采样概率

        self.base_image_dir = base_image_dir = LISA_ROOT
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.num_datasets = len(self.datasets)

        self.num_be_called = 0
        if balance_sample:
            self.dataset_sample_list = get_random_list(probabilities=self.sample_rate.tolist(), values=list(range(self.num_datasets)), length=samples_per_epoch)   # 为一个epoch中的每个数据确定属于哪个数据集
            chatunivi_sample_range = [int(i) for i in univi_sample_frame_range.split(',')]     # [8, 12]
            chatunivi_range_length = chatunivi_sample_range[-1] - chatunivi_sample_range[0] + 1      # 5
            self.chatunivi_sample_list = get_random_list(probabilities=[float(1 / chatunivi_range_length) for _ in range(chatunivi_range_length)], values=list(range(chatunivi_sample_range[0],chatunivi_sample_range[-1]+1)), length=10000) # 10000项等概率数组
            rvos_sample_range = [int(i) for i in rvos_num_frames_sample_range.split(',')]      # [6, 12]
            rvos_range_length = rvos_sample_range[-1] - rvos_sample_range[0] + 1      # 7
            # [1/7, 1/7, ..., 1/7, 1/7]   对于子数据集的选择列表
            self.rvos_sample_list = get_random_list(probabilities=[float(1/rvos_range_length) for _ in range(rvos_range_length)], values=list(range(rvos_sample_range[0],rvos_sample_range[-1]+1)), length=10000)
        else:
            self.dataset_sample_list = None
            self.chatunivi_sample_list = []
            self.rvos_sample_list = []


        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(            # 语义分割数据集
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(         # refer_seg数据集
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(            # vqa数据集
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val, # new
                        reason_seg_data,
                        explanatory,
                    )
                )
            elif dataset == "reason_seg_sar":
                self.all_datasets.append(
                    SurgRSDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val, # new
                        reason_seg_data,
                    )
                )
            elif dataset == "chatunivi":
                self.all_datasets.append(
                    ChatUniviDataset(
                        tokenizer                = tokenizer,
                        vision_tower             = vision_tower,
                        samples_per_epoch        = samples_per_epoch,
                        precision                = precision,
                        image_size               = image_size,
                        univi_data_list          = univi_data_list,
                        univi_data_ratio         = univi_data_ratio,
                        univi_max_image_len      = univi_max_image_len,
                        image_aspect_ratio       = 'pad',
                        univi_sample_frame_range = univi_sample_frame_range,
                        univi_sample_list        = self.chatunivi_sample_list,
                    )
                )
            elif dataset == "rvos":
                self.all_datasets.append(
                    RVOSDataset(
                        tokenizer                = tokenizer,
                        vision_tower             = vision_tower,
                        samples_per_epoch        = samples_per_epoch,
                        precision                = precision,
                        image_size               = image_size,
                        num_classes_per_sample   = num_classes_per_sample,
                        num_frames_sample_range  = rvos_num_frames_sample_range,     # 视频采样帧范围
                        rvos_sample_policy       = rvos_sample_policy,
                        rvos_seg_data            = rvos_seg_data,
                        rvos_sample_ratio        = rvos_sample_ratio,          # 各个ref-vos数据集采样比例
                        rvos_sample_list         = self.rvos_sample_list,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        self.num_be_called += 1
        if self.dataset_sample_list == None:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        else:
            ind = self.dataset_sample_list[self.num_be_called % self.samples_per_epoch] # 导致每一轮次的数据集排序情况是一致的
        data = self.all_datasets[ind]    # 读取数据集
        inference = False     # 读取其中的数据
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean   = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std    = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size     = 1024
    ignore_label = 255

    def __init__(
        self,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        use_sira_tokens=False,
    ):
        self.base_image_dir = LISA_ROOT
        self.use_sira_tokens = use_sira_tokens
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            if ds == "SurgRS":
                self.base_image_dir = os.path.join(self.base_image_dir, ds)
                # json_file_name = os.path.join(self.base_image_dir, "MMR", json_name)
                json_file_name = os.path.join(self.base_image_dir, split)
                with open(json_file_name, "r") as f:
                    reason_file = json.load(f)
                self.reason_seg_data = reason_file["data"]
                self.data_type = "multi_part_reason_seg"
            else:
                images = glob.glob(
                    os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
                )
                self.images = images
                self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(osp.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        osp.join(self.base_image_dir, 'refer_seg'),
                        "images/saiapr_tc-12",
                        item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        osp.join(self.base_image_dir, 'refer_seg'),
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        elif self.data_type == "multi_part_reason_seg":
            return len(self.reason_seg_data)
        else:
            return len(self.images)

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
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            # 得到image的reference
            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # 读取图像
            is_sentence = False
        elif self.data_type == "multi_part_reason_seg":
            image_info = self.reason_seg_data[idx]
            # if "file_name" in image_info:
            #     image_root = os.path.join(
            #         self.base_image_dir, "refer_seg/images/mscoco/images"
            #     )
            #     image_path = os.path.join(image_root, image_info["file_name"])

            image_path = image_info["image_path"]

            # anns = image_info["annotations"]

            question = image_info["questions"]
            gt_answer = image_info["answers"]
            text_answers = image_info["text_answers"]

            image_path = os.path.join(self.base_image_dir, image_path)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = True
            sampled_sents = question
            sampled_answers = text_answers
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]     # 得到mask

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        questions = []
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            _seg = sampled_answers[i]
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    convert2imagesplit(UNIFIED_LONG_QUESTION_LIST[0].format(sent=text), 1),
                )
                if self.data_type == "multi_part_reason_seg" and self.use_sira_tokens:
                    response = _seg.replace("[SEG]", "<SEG>") + "<QUERY>" * 64
                else:
                    response = "{} while for the entire video it is [TAK].".format(_seg)
                conv.append_message(conv.roles[1], response)
                questions.append(UNIFIED_LONG_QUESTION_LIST[0].format(sent=text))
            else:
                conv.append_message(
                    conv.roles[0],
                    convert2imagesplit(UNIFIED_SHORT_QUESTION_LIST[0].format(sent=text), 1),
                )
                conv.append_message(conv.roles[1], "Sure, the segmentation result of each frame is [SEG], while for the entire video it is [TAK].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        elif self.data_type == "multi_part_reason_seg":
            masks = []
            category_ids = []
            for answer_list in gt_answer:
                for answer in answer_list:
                    rle = answer["segmentation"]
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    masks.append(m)
                    category_ids.append(answer["category_id"])
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        category_ids = torch.tensor(category_ids)
        inference = True

        return (
            image_path,
            image,
            torch.stack([image_clip],dim=0),
            conversations,
            masks,
            labels,
            resize,
            questions,
            sampled_sents,
            category_ids,
            inference,
        )
