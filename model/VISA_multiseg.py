import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

# from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

from .univi.model.language_model.llama import ChatUniViLlamaForCausalLM, ChatUniViLlamaModel

from sam2.build_sam import _build_sam2_
from model.univi.constants import IMAGE_TOKEN_INDEX

from model.query_transformer import QueryTransformer

import time
import numpy as np


def dice_loss(
    inputs   : torch.Tensor,
    targets  : torch.Tensor,
    num_masks: float,
    scale    : float =1000,
    eps      : float =1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def fin(x, name):
    if x is None: return
    if not torch.isfinite(x).all():
        print(f"[NaN] {name}  shape={tuple(x.shape)} dtype={x.dtype} "
            f"min={x[torch.isfinite(x)].min().item() if torch.isfinite(x).any() else 'NA'} "
            f"max={x[torch.isfinite(x)].max().item() if torch.isfinite(x).any() else 'NA'}")
        raise RuntimeError(name)

class VrshqMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VrshqMetaModel, self).__init__(config) # 意味着先调用父类初始化，因为上一个函数有两个父类，所以这里是调用ChatUniViLlamaModel的模型初始化函数

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            # 因为缺少一些设定，进入这个分支, 所以initialize_lisa_modules需要外部调用实现
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None) # sam2 large checkpoint
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = _build_sam2_(ckpt_path=self.vision_pretrained, video_inference=False)

        # frozen the image encoder weights
        for param in self.visual_model.parameters():
            param.requires_grad = False
        # like the visa-1 version, we only fine tune the mask decoder 仅训练解码器
        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        def build_fc():
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.0),
            )

        self.text_hidden_fcs = nn.ModuleList([
            build_fc(),
            build_fc()
        ])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.query_transformer = QueryTransformer(depth=4, seq_len=64, embed_dim=256)
        self.query_transformer.train()
        for param in self.query_transformer.parameters():
            param.requires_grad = True

        self.cross_attention = nn.ModuleList([CrossAttention(), CrossAttention()])
        self.cross_attention.train()
        for param in self.cross_attention.parameters():
            param.requires_grad = True


class VrshqModel(VrshqMetaModel, ChatUniViLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VrshqModel, self).__init__(config, **kwargs) # 优先调用父类初始化函数，只调用第一个，第二个的会融入第一个的调用中，是父类的父类

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

# VrshqForCausalLM
# ├── ChatUniViLlamaForCausalLM（HF 因果语言模型外壳：forward/generate/基础 loss 逻辑；内部原本也带 base model 与 lm_head）
# ├── self.model = VrshqModel（你真正用来产出 hidden_states + 承载分割模块的“内核模型”）
# │ ├── VrshqMetaModel（装配 LISA/SAM 相关模块的 mixin）
# │ │ ├── self.visual_model = SAM2（build_sam2 构建；encoder 冻结；train_mask_decoder=True 时仅 sam_mask_decoder 可训练）
# │ │ └── self.text_hidden_fcs = ModuleList([MLP])（Linear(4096→4096)+ReLU+Linear(4096→256)+Dropout；把 seg/track token 的 hidden 映射到 SAM embedding 空间）
# │ └── ChatUniViLlamaModel（Transformer LLM 主体：输出 logits 与 output_hidden_states；接收 images/images_clip 走 mm_vision_tower=CLIP ViT-L/14）
# ├── self.lm_head = nn.Linear(hidden_size→vocab_size)（类tokenid映射，在 VrshqForCausalLM 里重新定义/覆盖的 lm_head）
# │
# ├── 第二部分：forward 中“显式调用但不是 nn.Module 属性”的关键组件
# ├── mm_vision_tower = CLIP ViT-L/14（由 ChatUniVi 配置 mm_vision_tower / vision_tower 提供，多模态视觉特征给 LLM）
# ├── seg_token_idx / query_token_idx（定位 <SEG>/<QUERY> token）
# └── mask losses：sigmoid_ce_loss + dice_loss（训练时与 ce_loss 加权求和）
class VrshqForCausalLM(ChatUniViLlamaForCausalLM): # HF-style causal LM 封装类，负责：forward，generate，loss 计算，tokenizer 对齐，lm_head
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            # 进入这个分支，详见config即ChatUniVi的预训练配置
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True) # 这里默认值可能有错误，train.py的args和ChatUniViLlamaForCausalLM的预训练配置均为false，已通过train.py改为false
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14") # 和ChatUniVi的预训练配置保持一致
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)         # 1.0
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)     # 0.5
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)       # 2
            self.target_loss_weight = kwargs.pop("target_loss_weight", None) # None
            # config._attn_implementation = "flash_attention_2"
        else:
            config.mm_vision_tower = config.vision_tower


        self.seg_token_idx = kwargs.pop("seg_token_idx")     # 32000
        self.query_token_idx = kwargs.pop("query_token_idx", None)
        legacy_query_token_idx = kwargs.pop("image_token_idx", None)
        if self.query_token_idx is None:
            self.query_token_idx = legacy_query_token_idx
        self.track_token_idx = kwargs.pop("track_token_idx", None)
        self.num_query_tokens = kwargs.pop("num_query_tokens", 64)
        self.num_seg_token = kwargs.pop("seg_token_num")     # 1
        self.alpha = kwargs.pop("alpha")                     # 0.1

        super().__init__(config) # 模型外壳父类初始化，主要是前处理后处理以及模型调用规则

        self.model = VrshqModel(config, **kwargs) # 1.ChatUniVi模型主体初始化，（2.SAM2视觉模型初始化，3.文本→分割投影MLP初始化）后面两个在外部train.py初始化

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 语言模型输出头先使用默认大小，后面根据实际大小修改

        # self.conv_1d = nn.Conv1d(256, 256, kernel_size=64)


        # self.learnable_query = nn.Parameter(torch.randn(1, 64, 256), requires_grad=True)
        # self.learnable_query.ds_full_param = True  # Keep full param in DeepSpeed ZeRO
        # self.learnable_query.ds_persist = True

        # self.model_num_of_query = 64

        # Initialize weights and apply final processing
        # self.post_init()
        # self._init_custom_params()
        self.post_init()

    # def _init_custom_params(self):
    #     """Initialize custom parameters."""
    #     nn.init.normal_(self.learnable_query, mean=0.0, std=0.02)
    #     nn.init.normal_(self.conv_1d.weight, mean=0.0, std=0.02)
    #     nn.init.zeros_(self.conv_1d.bias)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.model.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def sample_cond_frame(self, target, sample_list, num_frame):
        index = sample_list.index(target)
        # print(index, sample_list)
        if index == 0:
            frame_ids_sam = sample_list[0: index + num_frame]
        elif index == len(sample_list) -1:
            frame_ids_sam = sample_list[index - num_frame + 1: ]
        else:
            frame_ids_sam = sample_list[index - num_frame // 2 : index + num_frame // 2 + 1]

        return frame_ids_sam, frame_ids_sam.index(index)

    def model_forward(
        self,
        images: torch.FloatTensor,    # list: num_conv, t, 3, h, w
        images_clip: torch.FloatTensor,      # list: num_conv, t, 3, h, w
        input_ids: torch.LongTensor,     # num_sentence in a batch, num tokens in a sentence
        labels: torch.LongTensor,      # num_sentence in a batch, num tokens in a sentence, 问题部分全部屏蔽
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,        # 由于每个batch的conversation数量不一致，因此需要设置offset来区分不同的conversation分别属于哪个batch
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        conversation_list: List[str],
        num_frame_list: List[int],
        num_conv_list: List[int],
        inference: bool = False,
        **kwargs,
    ):

        ### 图像/视频bs与对话数量的一致性检查
        batch_size = len(images)
        # image_embeddings = self.get_visual_embs(torch.cat(images, dim=0))    # vision encoder: bs, dim, h, w
        assert batch_size == len(offset) - 1
        for batch_idx in range(batch_size):         # 对于每一个batch size，确定同一batch中的每个视频对应几个conversation
            assert num_conv_list[batch_idx] == offset[batch_idx + 1] - offset[batch_idx]

        if inference:
            length = input_ids.shape[0]
            assert len(images_clip) == 1, f'Inference only supports one video, but got {len(images_clip)} videos.'
            images_clip = [        # num_conv = 1 * t, dim, h, w all the frames
                images_clip[0].unsqueeze(0).expand(length, -1, -1, -1, -1).contiguous().flatten(0, 1)
            ]

            output_i = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            torch.cuda.empty_cache()

            output_hidden_states = output_i.hidden_states
            output = None

            num_image_ori_token = (input_ids[0] == IMAGE_TOKEN_INDEX).sum()
            assert all(
                [
                    (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == num_image_ori_token for i in range(length)
                ]
            )
            token_add = 111 * num_image_ori_token # 图像token的个数在该模型中固定被111个token代替

            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx        # 将每个conversation token中的 <seg> token转换为True
            seg_token_mask = torch.cat([seg_token_mask,  torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(), ], dim=1, )          # num_sentences, ntokens + 1
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], token_add)).bool().cuda(), seg_token_mask], dim=1, )
            all_conv_seg_token_num = seg_token_mask.sum(dim=1).tolist()

            query_token_mask = input_ids[:, 1:] == self.query_token_idx
            query_token_mask = torch.cat([query_token_mask, torch.zeros((query_token_mask.shape[0], 1)).bool().cuda()], dim=1)
            query_token_mask = torch.cat([torch.zeros((query_token_mask.shape[0], token_add)).bool().cuda(), query_token_mask], dim=1)

        else:
            ### 扩展每个batch中的图像/视频特征，以匹配conversation数量，便于进行LLM前向传播
            images_clip_list = []
            for batch_idx in range(batch_size):
                bs_conv_num = num_conv_list[batch_idx]     # 该视频中的conversation数量，不使用offset这个不稳定因素，直接使用数组索引
                images_clip_i = images_clip[batch_idx].unsqueeze(0).expand(bs_conv_num, -1, -1, -1, -1).contiguous()      # num_conv, t, 3, h, w
                images_clip_list.append(images_clip_i)
            images_clip_list = [i.flatten(0, 1) for i in images_clip_list]     # list: num_conv * t, 3, h, w

            # 调用父类的forward方法，获取输出，一次可以同时处理不同图像的所有对话，图片对应好就好
            output = super().forward(
                images=images_clip_list,
                attention_mask=attention_masks,       # num_conv, tokens
                input_ids=input_ids,
                labels=labels,                        # labels仅控制需要计算loss的tokens是哪些，其他token的loss为0，前向能看到所有input_ids
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states        # tuple of tensor, num_layers + 1 层数个，L+1层, num_sentences, num_tokens, in_dim  [33*torch.Size([3, 209, 4096]), ...]

            # 该模型多输出一个labels，为添加过图片（111*tokens）之后的仅答案未遮蔽的targets
            seg_token_mask = output.labels[..., 1:] == self.seg_token_idx        # 将每个conversation token中的 <seg> token转换为True，32000
            # 因为预测是向后的，故第一位bos不在预测中，前面取1:，但预测长度与输入保持一致，故需要在后面添加一个值，这个值肯定不是seg，直接设为否即可。
            seg_token_mask = torch.cat([seg_token_mask,  torch.zeros((seg_token_mask.shape[0], 1), device=output.labels.device).bool(), ], dim=1)    # num_sentences, ntokens + 1
            all_conv_seg_token_num = seg_token_mask.sum(dim=1).tolist()

            query_token_mask = output.labels[..., 1:] == self.query_token_idx
            query_token_mask = torch.cat([query_token_mask, torch.zeros((query_token_mask.shape[0], 1), device=output.labels.device).bool()], dim=1)

        assert len(self.model.text_hidden_fcs) == 2

        # 找到[seg]位置的嵌入，token （id）相同，语义嵌入hidden_states不同；语义嵌入不同，引导分割输出自然不同。
        pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states[-1][seg_token_mask])       # [seg.num, 256] # num_sentences in the batch, out_dim
        seg_token_counts = seg_token_mask.int().sum(-1)       # [seg.num] # num_sentences in the batch   每个对话有几个seg_token, 原设定值全为1
        seg_token_offset = seg_token_counts.cumsum(-1)       # 累加值 # 这里需要注意：因为有的sentence中不含有seg_token，因此需要通过一定方式将所有的seg_token提取出来
        seg_token_offset_ = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        ) # [seg.num+1]
        seg_token_offset = seg_token_offset_[offset]     # [B+1] 从对话中进一步取值至bs，记录每张图像/视频的[seg]嵌入起始和结束位置

        query_token_counts = query_token_mask.int().sum(-1)
        if not torch.all(query_token_counts == self.num_query_tokens):
            raise ValueError(
                f"Each response must contain exactly {self.num_query_tokens} <QUERY> tokens, "
                f"got {query_token_counts.tolist()}"
            )
        pred_query_embeddings = torch.stack(
            [
                self.model.text_hidden_fcs[1](
                    output_hidden_states[-1][i][query_token_mask[i]]
                )
                for i in range(query_token_mask.shape[0])
            ],
            dim=0,
        )


        # pred_embeddings_ = []
        # pred_track_embeddings_ = []
        # for i in range(len(track_token_offset) - 1):
        #     start_i, end_i = track_token_offset[i], track_token_offset[i + 1]      # 得到属于各个视频的每个conversation对应的tak_token索引
        #     num_frame = len(images[i]) # T 第i个视频抽样帧数
        #     start_frame_i = 0 * num_frame # 该视频的起始抽样帧
        #     end_frame_i = (end_i - start_i) * num_frame   # 该视频的结束抽样帧（由tak设定为1得来）
        #     batch_pred_embeddings = pred_embeddings[start_frame_i: end_frame_i] # [conv.num * num_frame, dim] # 这里抛弃了seg_token_offset，因为个对话只有一个目标[seg]
        #     batch_pred_embeddings = batch_pred_embeddings.reshape(len(batch_pred_embeddings) // num_frame, num_frame, batch_pred_embeddings.shape[-1]) # [seg.num*conv.num, num_frame, dim]
        #     pred_embeddings_.append(batch_pred_embeddings)
        #     pred_embeddings = pred_embeddings[end_frame_i: ]

        #     pred_track_embeddings_.append(pred_track_embeddings[start_i: end_i]) # [tak.num*conv.num, dim]

        pred_embeddings_ = []
        pred_query_embeddings_ = []

        for i in range(batch_size):
            seg_s, seg_e = seg_token_offset[i].item(), seg_token_offset[i + 1].item()
            batch_seg = pred_embeddings[seg_s:seg_e]   # [Ki, dim], Ki is the number of <SEG> tokens.

            num_frame = len(images[i])  # 图像时=1
            batch_seg = batch_seg.view(batch_seg.shape[0], num_frame, batch_seg.shape[-1])  # [Ki, 1, dim]
            pred_embeddings_.append(batch_seg)

            conv_s, conv_e = offset[i].item(), offset[i + 1].item()
            batch_queries = pred_query_embeddings[conv_s:conv_e]
            batch_seg_counts = seg_token_counts[conv_s:conv_e]
            pred_query_embeddings_.append(
                torch.repeat_interleave(batch_queries, batch_seg_counts, dim=0)
            )

        pred_embeddings = pred_embeddings_        # list bs: num_conversations, t, dim
        pred_query_embeddings = pred_query_embeddings_

        assert len(pred_embeddings) == batch_size
        assert len(pred_query_embeddings) == batch_size

        # mask decoder
        pred_masks = []
        gt_masks = []
        alpha = self.alpha
        for i in range(batch_size):     # 训练阶段的batch size数量较大

            if len(images[i]) == 1:       # 图像数据，因为视频的帧数一定超过1

                ###### new, 进行seg token以及track token的融合
                # global_embedding = []
                # track_embeddings = pred_track_embeddings[i].unsqueeze(1)       # num_conv, 1, dim 先不调整这里的标注了
                # seg_embeddings = pred_embeddings[i]      # num_conv, t, dim
                # if track_embeddings.shape[0] != 0:
                #     for num_conv, seg_token in enumerate(track_embeddings):
                #         track_embed = track_embeddings[num_conv]       # 1, dim
                #         seg_embed = seg_embeddings[num_conv]       # t, dim
                #         # similarity = seg_embed @ track_embed.transpose(0, 1)       # t, 1
                #         # similarity = torch.sigmoid(similarity)     # t, 1

                #         # 对于图像和视频，都将各帧seg token按相似度和global seg token进行加权融合
                #         track_embed = track_embed # + alpha * similarity.transpose(0, 1) @ seg_embed      # 1, dim
                #         global_embedding.append(track_embed)
                #     global_embedding = torch.cat(global_embedding, dim=0)      # num_conv, dim
                # else:
                #     global_embedding = track_embeddings.squeeze(1)

                # 图片临时使用
                # box_end_embedding = self.get_sam_embedding(pred_image_embeddings[i], pred_embeddings)

                query_hidden_state = self.model.query_transformer(pred_query_embeddings[i])
                seg_embeddings = self.model.cross_attention[1](
                    x_q=pred_embeddings[i],
                    x_kv=query_hidden_state,
                )[0]
                global_embedding = seg_embeddings[:, 0, :]    # [Ki, dim], one prompt per <SEG>.

                ###### 视觉编码部分 vision backbone 这里在做get_visual_embs的sam2版本操作
                backbone_out = self.model.visual_model.forward_image(images[i])      # 1, 3, h, w
                _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)      # list: hw, 1, c

                # sam2：添加记忆注意力，对于图片需要添加默认无记忆嵌入（仅在最后一层）
                if self.model.visual_model.directly_add_no_mem_embed:    # 在第一阶段添加无记忆嵌入 add no memory embeddings to the first stage
                    # 成功进入该分支
                    vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed

                # 重新组织各层特征的形状，便于后续解码器使用
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)     # 1, c, h, w 对于每一层
                    for feat, feat_size in zip(vision_feats[::-1], self.model.visual_model._bb_feat_sizes[::-1])
                ][::-1]
                self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}      # feats[-1]: 最高级特征且分辨率最低，feats[:-1]: 两层低级特征，分辨率较高
                # print('self._features', self._features["image_embed"].shape, self._features["high_res_feats"][0].shape)

                # 视觉特征互相关
                Bq = query_hidden_state.shape[0]   # 3
                C  = self._features["image_embed"].shape[1]          # 256
                H, W = self._features["image_embed"].shape[2:]       # 64, 64

                # [1, 256, 64, 64] → [1, 4096, 256]
                image_tokens = (
                    self._features["image_embed"]
                    .flatten(2)          # [1, 256, 4096]
                    .transpose(1, 2)     # [1, 4096, 256]
                )
                # [1, 4096, 256] → [3, 4096, 256]
                image_tokens = image_tokens.expand(Bq, -1, -1).contiguous()
                att_image_tokens, _ = self.model.cross_attention[0](
                    x_q=image_tokens,         # [3, 4096, 256]
                    x_kv=query_hidden_state,  # [3, 64, 256]
                )
                att_image_embeddings = (
                    att_image_tokens
                    .transpose(1, 2)          # [3, 256, 4096]
                    .reshape(Bq, 256, 64, 64)
                )

                high_res_features = [feat_level[0].unsqueeze(0).to(global_embedding.dtype) for feat_level in self._features["high_res_feats"]]

                multimask_output = False      # 这里是否仅输出一个mask

                ###### 提示生成部分
                video_masks = []
                for target_idx in range(global_embedding.shape[0]):
                    global_embedding_ = global_embedding[target_idx:target_idx + 1]

                    sparse_embeddings_, dense_embeddings_ = self.model.visual_model.sam_prompt_encoder(    # num_conv, 0, dim       num_conv, 256, 64, 64
                                                                points=None,
                                                                boxes=None,
                                                                masks=None,
                                                                batch_size=global_embedding_.shape[0],    # n
                                                            ) # 下面将原本在sam_prompt_encoder中计算得到的sparse_embeddings和dense_embeddings调整到外面一致，不改变sam2结构
                    sparse_embeddings_ = torch.cat((sparse_embeddings_, global_embedding_.unsqueeze(1)), dim=1).to(global_embedding.dtype)     # num_conv, 0, dim ------> num_conv, 1, dim

                    if sparse_embeddings_.shape[0] == 0:
                        pass

                    ###### 解码与后处理，使用所有层特征进行解码，在医学上按说效果更好
                    low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=att_image_embeddings[target_idx].unsqueeze(0).to(global_embedding.dtype),
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings_,
                        dense_prompt_embeddings=dense_embeddings_,
                        multimask_output=multimask_output,
                        repeat_image=True, # sam2实现了1张图复制多次，sam版不需要设置会自动复制
                        high_res_features=high_res_features,
                    )

                    video_masks_ = self.model.visual_model._transforms.postprocess_masks(
                        masks=low_res_masks, orig_hw=(1024, 1024)
                    ) # 还原至训练尺寸，num_conv, 1, 1024, 1024
                    video_masks_ = video_masks_[..., : resize_list[i][0], : resize_list[i][1]] # 去除长边黑边padding
                    video_masks_ = F.interpolate(
                            video_masks_, label_list[i].shape, mode="bilinear", align_corners=False
                        )      # 1, 1, h, w 插值至gt尺寸
                    video_masks.append(video_masks_)  # [1,1,h,w]

                video_masks = torch.cat(video_masks, dim=0) # num_conv, t, h, w
                pred_masks.append(video_masks[:, 0])    # num_conv, h, w
                gt_masks.append(masks_list[i])       # num_conv, t, h, w这里是num_conv, t, h, w

            else:     # SIRA is defined for image-based SurgRS samples.
                raise NotImplementedError(
                    "The SIRA SEG/QUERY implementation supports image samples only. "
                    "Use the legacy VRS-HQ model for video datasets."
                )
                # 视频数据，帧数和送入MLLM中的一致
                # 根据真实的关键帧数据以及pred_target_logits的分数来选择关键帧送入SAM中

                video_mask_list_ = []
                gt_masks_list = []
                mask = masks_list[i]     # num_conv, t, h, w

                track_embeddings = pred_track_embeddings[i].unsqueeze(1)       # num_conv, 1, dim
                seg_embeddings = pred_embeddings[i]      # num_conv, t, dim
                sam_images_ = images[i]      # t, 3, h, w

                # 按照对话进行循环
                for num_conv, seg_token in enumerate(track_embeddings):

                    ###### new, 进行seg token以及track token(全局)的融合
                    track_embed = track_embeddings[num_conv]       # 1, dim
                    seg_embed = seg_embeddings[num_conv]       # t, dim
                    # similarity = seg_embed @ track_embed.transpose(0, 1)       # t, 1
                    # 不使用简单的相乘，而是使用余弦相似度进行融合
                    similarity = F.cosine_similarity(seg_embed, track_embed, dim=1).unsqueeze(1)
                    similarity = torch.softmax(similarity, dim=0)     # t, 1
                    # 对于图像和视频，都将各帧seg token按相似度和global seg token进行加权融合（相似度视为权重）
                    track_embed = track_embed + alpha * similarity.transpose(0, 1) @ seg_embed      # 1, dim

                    seg_index = torch.argmax(similarity[:, 0], dim=0)       # t 筛选出相似度最高的帧级seg token对应的帧序号
                    frame_ids_sam, cond_frame_idx = self.sample_cond_frame(seg_index, sample_list=list(range(len(sam_images_))), num_frame=3) # 选出关键帧周边的3个帧送入sam2
                    sam_images = torch.stack([sam_images_[id] for id in frame_ids_sam], dim=0)      # 3, 3, h, w
                    gt_masks_list.append(torch.stack([mask[num_conv][id] for id in frame_ids_sam]))     # 3, h, w

                    ###### sam2 视频功能准备（注意函数前有train的都是微调过的） initialize the sam2 for training
                    inference_state = self.model.visual_model.train_init_state(sam_images)    # 3, 3, h, w 基于 3 帧图像构建一个 state，里面可能缓存 backbone 特征、memory 容器、对象表等。
                    self.model.visual_model.reset_training_state(inference_state)       # 清空该视频旧的 memory / object 状态，保证这个目标从干净状态开始

                    # 训练阶段仅提供一个条件帧，这里对于不同目标输入的视频不同
                    _, out_obj_ids, out_mask_logits, _, _ = self.model.visual_model.train_add_new_points(
                            inference_state=inference_state,
                            frame_idx=cond_frame_idx,
                            obj_id=(num_conv + 1), # 这里一个对话只有一个对象，所以直接用 num_conv + 1
                            pred_embeddings=track_embed,        # 1, dim
                    )

                    if cond_frame_idx == 0:
                        video_mask_list = []
                        for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state,
                                                                                                                                start_frame_idx=0,
                                                                                                                                reverse=False):
                            masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                            masks = F.interpolate(
                                masks, label_list[i].shape, mode="bilinear", align_corners=False
                            )      # 1, 1, h, w
                            video_mask_list.append(masks[0, 0])     # h, w

                        video_masks = torch.stack(video_mask_list, dim=0)     # t, h, w

                    elif cond_frame_idx == len(sam_images) - 1:
                        video_mask_list = []
                        for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state,
                                                                                                                                start_frame_idx=cond_frame_idx,
                                                                                                                                reverse=True):      # bs=1, 1, h, w
                            masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                            masks = F.interpolate(
                                masks, label_list[i].shape, mode="bilinear", align_corners=False
                            )      # 1, 1, h, w
                            video_mask_list.append(masks[0, 0])     # h, w

                        video_masks = torch.stack(video_mask_list, dim=0)     # t, h, w
                        video_masks = torch.flip(video_masks, dims=[0])   # t, h, w
                        # video_masks = video_masks.transpose(0, 1)     # 3, t, h, w

                    else:
                        video_mask_list1 = []
                        video_mask_list2 = []
                        for reverse in [False, True]:
                            for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state,
                                                                                                                                    start_frame_idx=cond_frame_idx,
                                                                                                                                    reverse=reverse):
                                masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                                masks = F.interpolate(
                                    masks, label_list[i].shape, mode="bilinear", align_corners=False
                                )      # 1, 1, h, w

                                if reverse == False:
                                    video_mask_list1.append(masks[0, 0])
                                else:
                                    video_mask_list2.append(masks[0, 0])

                            # 再预测一遍条件帧信息（换向需要重新启动）
                            self.model.visual_model.reset_training_state(inference_state)
                            _, out_obj_ids, out_mask_logits, _, _ = self.model.visual_model.train_add_new_points(
                                    inference_state=inference_state,
                                    frame_idx=cond_frame_idx,
                                    obj_id=(num_conv + 1),
                                    pred_embeddings=track_embed,        # 1, dim
                            )

                        video_pred_masks1 = torch.stack(video_mask_list1, dim=0)     # t, h, w
                        video_pred_masks2 = torch.stack(video_mask_list2, dim=0)     # t, h, w
                        video_pred_masks2 = torch.flip(video_pred_masks2, dims=[0])     # t, h, w
                        video_masks = torch.cat((video_pred_masks2[: -1], video_pred_masks1), dim=0)     # T, h, w

                    video_mask_list_.append(video_masks)

                video_masks_ = torch.stack(video_mask_list_, dim=0)      # num_conv, 3, h, w
                pred_masks.append(video_masks_.flatten(0, 1))     # num_conv * 3, h, w

                gt_masks.append(torch.stack(gt_masks_list, dim=0))    # num_conv, 3, h, w

        model_output = output

        gt_masks = [mm.flatten(0, 1) for mm in gt_masks] # bs*[num_conv * 1/3, h, w] （是因为loss计算支持这个格式的一次计算）

        if inference:
            return {
                "pred_masks": video_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ###### 计算loss
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(batch_size):       # 损失计算
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0] # 不同图像/视频的对话
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )

            num_masks += gt_mask.shape[0] # 此时一个对话只有一个[seg] token

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def get_sam_embedding(self, query_hidden_state, seg_hidden_state, if_detach_res_loss=False):
        """
        Extract and project SAM embedding from the last learnable queries in hidden states.
        """
        # query_hidden_state = hidden_states[:, -self.model_num_of_query:]
        # query_hidden_state = pred_image_embeddings

        if if_detach_res_loss:
            query_hidden_state = query_hidden_state.detach()

        # Apply QTrans to the query-level token embeddings.
        out_hidden_state=[]
        query_hidden_state = self.model.query_transformer(query_hidden_state)


        for i in range(len(seg_hidden_state)):
            out_hidden_state.append( self.model.cross_attention(
                                        x_q=seg_hidden_state[i].transpose(0, 1),       # num_conv, t, dim (CrossAttention x_q: torch.Size([1, 10, 256]) x_kv: torch.Size([1, 64, 256]))
                                        x_kv=query_hidden_state[i].unsqueeze(0),      # num_conv, num_query, dim
                                    )[0].squeeze(0).unsqueeze(1) )    # num_conv, t, dim

        return out_hidden_state


    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")


from typing import Optional, Tuple

class CrossAttention(nn.Module):
    """
    Single-head cross-attention (no multi-head), all dims fixed to 256.

    Inputs:
      x_q  : [B, Lq, 256]  (queries)
      x_kv : [B, Lk, 256]  (keys/values from context)
      key_padding_mask: [B, Lk] bool, True means "mask out / ignore" that kv token
      attn_mask: [Lq, Lk] or [B, Lq, Lk] bool/float mask (optional)

    Output:
      y: [B, Lq, 256]
      attn (optional): [B, Lq, Lk]
    """
    def __init__(self, dropout: float = 0.0, use_bias: bool = True, add_ffn: bool = True):
        super().__init__()
        self.dim = 256
        self.dropout = nn.Dropout(dropout)
        self.add_ffn = add_ffn

        self.norm_q = nn.LayerNorm(256)
        self.norm_kv = nn.LayerNorm(256)

        self.q_proj = nn.Linear(256, 256, bias=use_bias)
        self.k_proj = nn.Linear(256, 256, bias=use_bias)
        self.v_proj = nn.Linear(256, 256, bias=use_bias)
        self.out_proj = nn.Linear(256, 256, bias=use_bias)

        self.norm_out = nn.LayerNorm(256)

        if add_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(256, 256, bias=use_bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 256, bias=use_bias),
                nn.Dropout(dropout),
            )
            self.norm_ffn = nn.LayerNorm(256)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert x_q.dim() == 3 and x_kv.dim() == 3, "x_q and x_kv must be [B, L, 256]"
        assert x_q.size(-1) == 256 and x_kv.size(-1) == 256, "last dim must be 256"

        target_dtype = self.norm_q.weight.dtype
        if x_q.dtype != target_dtype:
            x_q = x_q.to(target_dtype)
        if x_kv.dtype != target_dtype:
            x_kv = x_kv.to(target_dtype)

        # Pre-norm
        q_in = self.norm_q(x_q)    # [B, Lq, 256]
        kv_in = self.norm_kv(x_kv) # [B, Lk, 256]

        # Projections
        q = self.q_proj(q_in)      # [B, Lq, 256]
        k = self.k_proj(kv_in)     # [B, Lk, 256]
        v = self.v_proj(kv_in)     # [B, Lk, 256]

        # Attention scores: [B, Lq, Lk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (256 ** 0.5)

        # Mask kv tokens: key_padding_mask True => mask out
        if key_padding_mask is not None:
            mask = key_padding_mask.to(torch.bool)[:, None, :]  # [B, 1, Lk]
            scores = scores.masked_fill(mask, float("-inf"))

        # Optional attn_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)         # [B, Lq, Lk]
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)               # [B, Lq, 256]
        y = self.out_proj(y)                    # [B, Lq, 256]

        # Residual + norm
        y = self.norm_out(x_q + y)

        # Optional FFN (keeps 256 throughout)
        if self.add_ffn:
            y = self.norm_ffn(y + self.ffn(y))

        return (y, attn) if return_attn else (y, None)
