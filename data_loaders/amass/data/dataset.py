""" Code adapted from https://github.com/c-he/NeMF"""
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2019.08.09

# len(train_dataset) = 11642, len(test_dataset) = 164, len(valid_dataset) = 1668
# train_dataset.ds is a dictionary with keys = (['trans', 'rotmat', 'pos', 'angular', 'contacts', 'height', 'root_vel', 'velocity', 'global_xform', 'root_orient', 'rot6d'])
# train_dataset.ds['pos'] is a tensor of shape [11642, 128, 24, 3]

import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '..'))

import glob
import torch
from torch.utils.data import Dataset

class AMASS(Dataset):
    """AMASSN: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""
    """Adopted from NeMF codebase: https://github.com/c-he/NeMF """

    def __init__(self, split='train'):
        self.root_dir = 'dataset/amass/generative'
        self.dataset_dir = os.path.join(self.root_dir, split)
        self.ds = {}
        for data_fname in glob.glob(os.path.join(self.dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).split('-')[0]
            self.ds[k] = torch.load(data_fname)
        self.clip_length = 128
        self.mean = torch.load(os.path.join(self.root_dir, 'mean-male-128-30fps.pt')) # [1, 1, clip_length, dim]
        self.std = torch.load(os.path.join(self.root_dir, 'std-male-128-30fps.pt')) # [1, 1, clip_length, dim]

    def __len__(self):
        return len(self.ds['trans'])

    def _normalize_field(self, value, key):
        return (value - self.mean[key][0]) / self.std[key][0]

    def _denormalize_field(self, value, key):
        return value * self.std[key][0] + self.mean[key][0]

    def normalize(self, data):
        data = data.copy()
        for key in data.keys():
            data[key] = self._normalize_field(data[key], key)
        return data

    def denormalize(self, data):
        data = data.copy()
        for key in data.keys():
            data[key] = self._denormalize_field(data[key], key)
        return data

    def __getitem__(self, idx):
        data = []
        for key in self.ds.keys():
            # value is of shape [datset_size, clip_length, dim]
            normalized_item = self._normalize_field(self.ds[key][idx], key)
            data.append(normalized_item.reshape(self.clip_length, -1))
        data = torch.cat(data, dim=-1)
        return data