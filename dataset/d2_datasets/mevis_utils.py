###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################


import json
import logging
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from collections import defaultdict
"""
This file contains functions to parse MeViS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

def load_mevis_json(image_root, json_file, dataset_name, is_train: bool = False):

    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']     # 加载某一视频类型中的expression, 如ReVOS/train 中的所有视频表达式
    videos = list(subset_expressions_by_video.keys())  # 一共416个验证集视频，每个视频对应的exp以及各个exp对应的帧, 例如["OVIS/train/00e92ab4", ...]
    print('number of video in the datasets:{}'.format(len(videos)))
    metas = []
    is_train = (image_root.split('/')[-1] == 'train') or is_train
    if is_train:
        # 打开mask_dict.json
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        vid2metaid = defaultdict(list)      # 视频名称：标注数量
        for vid in videos:  # d56a6ec78cfa, 377b1c5f365c, ...   对于每一个视频
            # vid_data    = {'expressions': dict, 'vid_id': int, 'frames': List[int], "height": int, "width": int}
            # expressions = {'0': {"exp": str, "obj_id": List[int], "anno_id": List[int]}, ...}
            vid_data   = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
            vid_len    = len(vid_frames)
            if vid_len < 2: # 单张视频算作图片，不在考虑范围内
                continue
            # if ('rgvos' in dataset_name) and vid_len > 80:
            #     continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video']    = vid  # 当前视频编号, 对于该数据集唯一 377b1c5f365c
                meta['exp']      = exp_dict['exp']  # 当前问题表达 4 lizards moving around
                meta['obj_id']   = [int(x) for x in exp_dict['obj_id']]   # 当前问题表达指代物体编号 [0, 1, 1, 2, ...]
                meta['anno_id']  = [str(x) for x in exp_dict['anno_id']]  # 当前问题表达指代物体掩码编码, 对于该数据集唯一 [1000000, 1000001, 1000001, 1000002, ...]
                meta['frames']   = vid_frames  # 当前视频 ['00000', '00001', ...]
                meta['exp_id']   = exp_id  # 当前问题表达编号 '0'
                meta['category'] = 0 # 不是参考分割，永远是0
                meta['length']   = vid_len # 视频帧长度
                metas.append(meta)
                vid2metaid[vid].append(len(metas) - 1) # {'377b1c5f365c': [0, 1, 2, ...], ...}
    else:
        mask_dict = dict()
        vid2metaid = defaultdict(list)
        for vid in videos:     # 对于每个视频
            vid_data   = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len    = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():    # 对于每一条exp
                meta = {}
                meta['video']    = vid             # video_name
                meta['exp']      = exp_dict['exp']      # expression
                meta['obj_id']   = -1
                meta['anno_id']  = -1
                meta['frames']   = vid_frames      # 该exp对应的视频帧
                meta['exp_id']   = exp_id          # expression id
                meta['category'] = 0
                meta['length']   = vid_len         # n_video
                if 'tp' in exp_dict.keys():
                   meta['tp']    = exp_dict['tp']
                metas.append(meta)
                vid2metaid[vid].append(len(metas) - 1)
    return metas, mask_dict, vid2metaid, is_train