# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import os
import json
import numpy as np
import torch
import random
import core.utils as utils
import torch.utils.data as data

# 自定义数据集类，继承自torch.utils.data.Dataset
class NpyFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, class_dict, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)

        # 初始化参数
        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments

        if self.modal == 'all':
            # RGB和Flow特征路径
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            # 指定模态的特征路径
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        # 视频列表文件路径
        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        print('=> {} set has {} videos'.format(mode, len(self.vid_list)))

        # 注释文件路径
        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = class_dict
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    # 获取视频数据
    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            # 加载RGB和Flow特征
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            # 根据采样方式进行采样
            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            # 合并特征
            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            # 加载指定模态的特征
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            # 根据采样方式进行采样
            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    # 获取视频标签和临时注释
    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            # 标签
            label[self.class_name_to_idx[_anno['label']]] = 1
            # 类别注释
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_seg, self.num_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    # 注释的开始和结束时间
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    # 时间转换为帧数
                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)


    # 随机扰动采样
    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    # 均匀采样
    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)
