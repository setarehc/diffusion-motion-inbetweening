import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.common.quaternion import qinv, qrot
from data_loaders.humanml.scripts.motion_process import recover_from_ric, extract_features
from data_loaders.humanml.utils.paramUtil import *
from data_loaders.humanml.common.skeleton import Skeleton


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3:4 +
                                            (joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num *
                3] = std[4 + (joints_num - 1) * 9:4 +
                         (joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 +
                joints_num * 3:] = std[4 + (joints_num - 1) * 9 +
                                       joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data[
            'text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''


class Text2MotionDatasetV2(data.Dataset):
    """
    Args:
        std_multiplier: multiply the std by this value; maybe useful for diffusion models by keeping the range of data managable
    """
    def __init__(self,
                 opt,
                 mean,
                 std,
                 split_file,
                 w_vectorizer,
                 use_rand_proj=False,
                 proj_matrix_dir=None,
                 traject_only=False,
                 mode='train',
                 random_proj_scale=10.0,
                 augment_type='none',
                 std_scale_shift=(1., 0.),  # Test random projection
                 drop_redundant=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        self.use_rand_proj = use_rand_proj
        self.traject_only = traject_only
        self.mode = mode

        self.augment_type = augment_type
        assert self.augment_type in ['none', 'rot', 'full']

        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # NOTE: Small data for debugging
        # print(' --- Using small data for debugging ---')
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                # if True:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)

                if flag:
                    motion = motion[:opt.max_motion_length]
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        if use_rand_proj:
            self.init_random_projection(proj_matrix_dir,
                                        scale=random_proj_scale)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def get_std_mean(self, traject_only=None, drop_redundant=None):
        if traject_only is None:
            traject_only = self.traject_only
        if drop_redundant is None:
            drop_redundant = self.drop_redundant

        if traject_only:
            std = self.std[:4]
            mean = self.mean[:4]
        elif drop_redundant:
            std = self.std[:67]
            mean = self.mean[:67]
        else:
            std = self.std
            mean = self.mean
        std = std * self.std_scale_shift[0] + self.std_scale_shift[1]
        return std, mean

    def inv_transform(self, data, traject_only=None):
        if self.use_rand_proj:
            data = self.inv_random_projection(data)
        std, mean = self.get_std_mean(traject_only)
        return data * std + mean

    def inv_transform_th(self, data, traject_only=None, use_rand_proj=None):
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.inv_random_projection(data, mode="th")
        std, mean = self.get_std_mean(traject_only)
        return data * torch.from_numpy(std).to(
            data.device) + torch.from_numpy(mean).to(data.device)

    def transform_th(self, data, traject_only=None, use_rand_proj=None):
        std, mean = self.get_std_mean(traject_only)
        data = (data - torch.from_numpy(mean).to(
            data.device)) / torch.from_numpy(std).to(data.device)
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.random_projection(data, mode="th")
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data[
            'text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        # text_data = text_list[0] # for rebuttal experiments
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length -
                        1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length //
                        self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # NOTE: if used for training trajectory model, discard all but the first 4 values
        if self.traject_only:
            motion = motion[:, :4]

        if self.augment_type in ['full', 'rot']:
            # motion [length, 4 or 263]
            # Random rotation
            rand_rot = (torch.rand(1, 1) * 2.0 -
                        1.0) * np.pi / 4.  # Rand [-1,1)
            r_rot_quat = torch.zeros(1, 4)
            r_rot_quat[..., 0] = torch.cos(rand_rot)
            r_rot_quat[..., 2] = torch.sin(rand_rot)
            r_rot_quat = r_rot_quat.repeat(motion.shape[:-1] + (1, ))
            motion[:, 0:1] = motion[:, 0:1] + rand_rot.numpy()

            pos = torch.zeros(motion.shape[:-1] + (3, ))
            pos[..., [0, 2]] = torch.from_numpy(motion[..., 1:3])
            pos = qrot(qinv(r_rot_quat), pos)
            motion[:, [1, 2]] = pos[:, [0, 2]].numpy()

            # Random translation. Only care about (x,z)
            if self.augment_type == 'full':
                trans_size = 3.
                rand_trans = np.random.rand(1, 2) * 2.0 - 1.0  # Rand [-1,1)
                rand_trans = rand_trans * trans_size
                motion[:, [1, 2]] = motion[:, [1, 2]] + rand_trans

        if self.drop_redundant:
            # Only keep the first 4 values and 21 joint locations
            assert not self.use_rand_proj
            motion = motion[:, :67]

        "Z Normalization"
        std, mean = self.get_std_mean()
        motion = (motion - mean) / std

        # Projection
        # NOTE: Do not do random projection if mode is eval or gt
        if (not self.mode in ["eval", "gt"]) and self.use_rand_proj:
            # t x 263
            motion = self.random_projection(motion)

        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ],
                                    axis=0)

        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(
            tokens)

    def init_random_projection(self, save_at, scale: float):
        if os.path.isfile(os.path.join(save_at, "rand_proj.npy")):
            print(f"Loading random projection matrix from {save_at}")
            self.proj_matrix = np.load(os.path.join(save_at, "rand_proj.npy"))
            self.inv_proj_matrix = np.load(
                os.path.join(save_at, "inv_rand_proj.npy"))
        else:
            print(f"Creating random projection matrix {scale}")
            self.proj_matrix = torch.normal(
                mean=0, std=1.0, size=(263, 263),
                dtype=torch.float)  # / np.sqrt(263)

            # scale first three values (rot spd, x spd, z spd)
            self.proj_matrix[[0, 1, 2], :] *= scale
            self.proj_matrix = self.proj_matrix / np.sqrt(263 - 3 +
                                                          3 * scale**2)
            self.inv_proj_matrix = torch.inverse(self.proj_matrix)

            self.proj_matrix = self.proj_matrix.detach().cpu().numpy()
            self.inv_proj_matrix = self.inv_proj_matrix.detach().cpu().numpy()

        self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
        self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)

        np.save(os.path.join(save_at, "rand_proj.npy"), self.proj_matrix)
        np.save(os.path.join(save_at, "inv_rand_proj.npy"),
                self.inv_proj_matrix)

    def random_projection(self, motion, mode="np"):
        if mode == "th":
            return torch.matmul(motion, self.proj_matrix_th.to(motion.device))
        return np.matmul(motion, self.proj_matrix)

    def inv_random_projection(self, data, mode="np"):
        if mode == "th":
            return torch.matmul(data, self.inv_proj_matrix_th.to(data.device))
        return np.matmul(data, self.inv_proj_matrix)


'''For use of training baseline'''


class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data[
            'text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([
                src_motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ],
                                        axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3:4 +
                                            (joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num *
                3] = std[4 + (joints_num - 1) * 9:4 +
                         (joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 +
                joints_num * 3:] = std[4 + (joints_num - 1) * 9 +
                                       joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    '%s/%s' % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    'caption': line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN'
                    or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):
    """
    Args:
        std_multiplier: multiply the std by this value; maybe useful for diffusion models by keeping the range of data managable
    """
    def __init__(self,
                 opt,
                 mean,
                 std,
                 split_file,
                 use_rand_proj=False,
                 proj_matrix_dir=None,
                 traject_only=False,
                 std_scale_shift=(1., 0.),
                 drop_redundant=False):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        self.use_rand_proj = use_rand_proj
        if use_rand_proj:
            self.init_random_projection(proj_matrix_dir)
        self.traject_only = traject_only
        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text': [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def get_std_mean(self, traject_only=None, drop_redundant=None):
        if traject_only is None:
            traject_only = self.traject_only
        if drop_redundant is None:
            drop_redundant = self.drop_redundant
        if traject_only:
            std = self.std[:4]
            mean = self.mean[:4]
        elif drop_redundant:
            std = self.std[:67]
            mean = self.mean[:67]
        else:
            std = self.std
            mean = self.mean
        std = std * self.std_scale_shift[0] + self.std_scale_shift[1]
        return std, mean

    def inv_transform(self, data, traject_only=None, use_rand_proj=None):
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.inv_random_projection(data)
        std, mean = self.get_std_mean(traject_only)
        return data * std + mean

    def inv_transform_th(self, data, traject_only=None, use_rand_proj=None):
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.inv_random_projection(data, mode="th")
        std, mean = self.get_std_mean(traject_only)
        return data * torch.from_numpy(std).to(
            data.device) + torch.from_numpy(mean).to(data.device)

    def transform_th(self, data, traject_only=None, use_rand_proj=None):
        std, mean = self.get_std_mean(traject_only)
        data = (data - torch.from_numpy(mean).to(
            data.device)) / torch.from_numpy(std).to(data.device)
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.random_projection(data, mode="th")
        return data

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling

    def init_random_projection(self, save_at):
        if os.path.isfile(os.path.join(save_at, "rand_proj.npy")):
            self.proj_matrix = np.load(os.path.join(save_at, "rand_proj.npy"))
            self.inv_proj_matrix = np.load(
                os.path.join(save_at, "inv_rand_proj.npy"))
            self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
            self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)
        else:
            print("... No projection matrix ...")
            assert False

    def random_projection(self, motion, mode="np"):
        if mode == "th":
            return torch.matmul(motion, self.proj_matrix_th.to(motion.device))
        return np.matmul(motion, self.proj_matrix)

    def inv_random_projection(self, data, mode="np"):
        if mode == "th":
            return torch.matmul(data, self.inv_proj_matrix_th.to(data.device))
        return np.matmul(data, self.inv_proj_matrix)


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self,
                 mode,
                 datapath='./dataset/humanml_opt.txt',
                 split="train",
                 use_abs3d=False,
                 traject_only=False,
                 use_random_projection=False,
                 random_projection_scale=None,
                 augment_type='none',
                 std_scale_shift=(1., 0.),
                 drop_redundant=False,
                 num_frames=None,
                 **kwargs):
        self.mode = mode

        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        # TODO: modernize get_opt
        opt = get_opt(dataset_opt_path, device, mode, use_abs3d=use_abs3d, max_motion_length=num_frames)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        self.absolute_3d = use_abs3d
        self.traject_only = traject_only
        self.use_rand_proj = use_random_projection
        self.random_proj_scale = random_projection_scale
        self.augment_type = augment_type
        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        if self.use_rand_proj:
            if self.random_proj_scale == 10:
                # NOTE: legacy code
                proj_matrix_dir = "./dataset"
            else:
                proj_matrix_dir = os.path.join(
                    f'save/random_proj_{self.random_proj_scale:.0f}')
                os.makedirs(proj_matrix_dir, exist_ok=True)
            print(f'proj_matrix_dir = {proj_matrix_dir}')
        else:
            proj_matrix_dir = None

        ###
        print("mode =", mode)

        if self.absolute_3d:
            # If mode is 'gt' or 'eval', we will load the *original* dataset. Not the absolute rot, x, z.
            if mode == 'gt':
                # used by T2M models (including evaluators)
                self.mean = np.load(
                    pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
                self.std = np.load(
                    pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            # elif mode == :
            #     # used by MDM models
            #     self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            #     self.std = np.load(pjoin(opt.data_root, 'Std.npy'))
            elif mode in ['train', 'eval', 'text_only']:
                '''
                The 'eval' is here because we want inv_transform to work the same way at inference for model with abs3d,
                regradless of which dataset is loaded.
                '''
                # used by absolute model
                self.mean = np.load(pjoin(opt.data_root, 'Mean_abs_3d.npy'))
                self.std = np.load(pjoin(opt.data_root, 'Std_abs_3d.npy'))

            self.mean_gt = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_gt = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            self.mean_rel = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std_rel = np.load(pjoin(opt.data_root, 'Std.npy'))
            self.mean_abs = np.load(pjoin(opt.data_root, 'Mean_abs_3d.npy'))
            self.std_abs = np.load(pjoin(opt.data_root, 'Std_abs_3d.npy'))
        elif mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(
                pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            assert self.random_proj_scale == 10, 'mode text only support only random projection scale 10'
            print(
                f't2m dataset aug: {self.augment_type} std_scale_shift: {self.std_scale_shift}'
            )
            print(f't2m dataset drop redundant information: {self.drop_redundant}')
            self.t2m_dataset = TextOnlyDataset(
                self.opt,
                self.mean,
                self.std,
                self.split_file,
                use_rand_proj=self.use_rand_proj,
                proj_matrix_dir=proj_matrix_dir,
                traject_only=self.traject_only,
                std_scale_shift=self.std_scale_shift,
                drop_redundant=self.drop_redundant,)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'),
                                               'our_vab')
            print(
                f't2m dataset aug: {self.augment_type} std_scale_shift: {self.std_scale_shift}'
            )
            print(f't2m dataset drop redundant information: {self.drop_redundant}')
            self.t2m_dataset = Text2MotionDatasetV2(
                self.opt,
                self.mean,
                self.std,
                self.split_file,
                self.w_vectorizer,
                use_rand_proj=self.use_rand_proj,
                proj_matrix_dir=proj_matrix_dir,
                traject_only=self.traject_only,
                mode=mode,
                random_proj_scale=self.random_proj_scale,
                augment_type=self.augment_type,
                std_scale_shift=self.std_scale_shift,
                drop_redundant=self.drop_redundant,)
            # End test
            self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

        # Load necessay variables for converting raw motion to processed data
        data_dir = './dataset/000021.npy'
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.kinematic_chain = t2m_kinematic_chain
        # Get offsets of target skeleton
        example_data = np.load(data_dir)
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        # (joints_num, 3)
        tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

    def motion_to_rel_data(self, motion, model):
        motion_bu = motion.detach().clone()
        # Right/Left foot
        fid_r, fid_l = [8, 11], [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx = [2, 1, 17, 16]
        sample_rel_np_list = []
        for ii in range(len(motion)):
            # Data need to be [120 (timestep), 22, 3] to get feature
            sample_rel = extract_features(
                motion[ii].detach().cpu().clone().permute(2, 0,
                                                          1).cpu().numpy(),
                0.002, self.n_raw_offsets, self.kinematic_chain,
                face_joint_indx, fid_r, fid_l)
            # Duplicate last motion step to match the size
            sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
            sample_rel = torch.cat(
                [sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1)
            # Normalize with relative normalization
            sample_rel = (sample_rel - self.mean_rel) / self.std_rel
            sample_rel = sample_rel.unsqueeze(1).permute(0, 3, 1, 2)
            sample_rel = sample_rel.to(motion.device)
            sample_rel_np_list.append(sample_rel)

        processed_data = torch.cat(sample_rel_np_list, axis=0)

        # NOTE: check if the sequence is still that same after extract_features and converting back
        # sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(0, 2, 3, 1)).float()
        # sample_after = (processed_data.permute(0, 2, 3, 1) * self.std_rel) + self.mean_rel
        # n_joints = 22
        # sample_after = recover_from_ric(sample_after, n_joints, abs_3d=False)
        # sample_after = sample_after.view(-1, *sample_after.shape[2:]).permute(0, 2, 3, 1)

        # rot2xyz_pose_rep = 'xyz'
        # rot2xyz_mask = None
        # sample_after = model.rot2xyz(x=sample_after,
        #                     mask=rot2xyz_mask,
        #                     pose_rep=rot2xyz_pose_rep,
        #                     glob=True,
        #                     translation=True,
        #                     jointstype='smpl',
        #                     vertstrans=True,
        #                     betas=None,
        #                     beta=0,
        #                     glob_rot=None,
        #                     get_rotations_back=False)

        # from data_loaders.humanml.utils.plot_script import plot_3d_motion
        # plot_3d_motion("./test_positions_1.mp4", self.kinematic_chain, motion[2].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
        # plot_3d_motion("./test_positions_1_after.mp4", self.kinematic_chain, sample_after[2].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

        # Return data already normalized with relative mean and std. shape [bs, 263, 1, 120(motion step)]
        return processed_data


    def motion_to_abs_data(self, motion, model):
        """
        Follows how abs3d dataset is initially created.
        First, create the relative data, then compute the absolute root rot/pos from it, and replace it into the relative date.
        """
        motion_bu = motion.detach().clone() # [bs, 22, 3, 196]
        # Right/Left foot
        fid_r, fid_l = [8, 11], [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx = [2, 1, 17, 16]
        sample_abs_np_list = []
        for ii in range(len(motion)):
            # Data need to be [120 (timestep), 22, 3] to get feature
            sample_rel = extract_features(
                motion[ii].detach().cpu().clone().permute(2, 0,
                                                          1).cpu().numpy(),
                0.002, self.n_raw_offsets, self.kinematic_chain,
                face_joint_indx, fid_r, fid_l)
            # Duplicate last motion step to match the size
            sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
            sample_rel = torch.cat([sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1) # [1, 196, 263]
            # Compute absolute root information instead of relative
            from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos
            r_rot_quat, r_pos, rot_ang = recover_root_rot_pos(sample_rel[None], abs_3d=False, return_rot_ang=True)
            sample_abs = sample_rel[None].clone()
            sample_abs[..., 0] = rot_ang
            sample_abs[..., [1, 2]] = r_pos[..., [0, 2]]
            # Normalize with absolute normalization
            sample_abs = (sample_abs - self.mean_abs) / self.std_abs # TODO: Check if correct stats are used
            sample_abs = sample_abs.permute(0, 3, 1, 2)
            sample_abs = sample_abs.to(motion.device)
            sample_abs_np_list.append(sample_abs)

        processed_data = torch.cat(sample_abs_np_list, axis=0)

        return processed_data


# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self,
                 mode,
                 datapath='./dataset/kit_opt.txt',
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)


def sample_to_motion(sample_abs, dataset, model):
    n_joints = 22
    # (bs, 263, 1, 120)
    # In case of random projection, this already includes undoing the random projection
    sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(
        0, 2, 3, 1)).float()

    sample = recover_from_ric(sample, n_joints, abs_3d=True)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    rot2xyz_pose_rep = 'xyz'
    rot2xyz_mask = None
    sample = model.rot2xyz(x=sample,
                           mask=rot2xyz_mask,
                           pose_rep=rot2xyz_pose_rep,
                           glob=True,
                           translation=True,
                           jointstype='smpl',
                           vertstrans=True,
                           betas=None,
                           beta=0,
                           glob_rot=None,
                           get_rotations_back=False)
    return sample


def abs3d_to_rel(sample_abs, dataset, model):
    '''We want to change the first 3 values from absolute to relative
    sample_abs shape [bs, 263, 1, 196]
    '''
    n_joints = 22
    # (bs, 263, 1, 120)
    # In case of random projection, this already includes undoing the random projection
    sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(
        0, 2, 3, 1)).float()

    sample = recover_from_ric(sample, n_joints, abs_3d=True)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    rot2xyz_pose_rep = 'xyz'
    rot2xyz_mask = None
    sample = model.rot2xyz(x=sample,
                           mask=rot2xyz_mask,
                           pose_rep=rot2xyz_pose_rep,
                           glob=True,
                           translation=True,
                           jointstype='smpl',
                           vertstrans=True,
                           betas=None,
                           beta=0,
                           glob_rot=None,
                           get_rotations_back=False)

    # sample now shape [32, 22, 3, 196].
    # from data_loaders.humanml.utils.plot_script import plot_3d_motion
    # plot_3d_motion("./test_positions_1.mp4", dataset.kinematic_chain, sample[4].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

    # Now convert skeleton back to sample with relative representation
    sample_rel = dataset.motion_to_rel_data(sample, model)

    return sample_rel


def rel_to_abs3d(sample_rel, dataset, model):
    """We want to change the first 3 values from relative to absolute

    Args:
        sample_rel (torch.tensor): shape [bs, 263, 1, 196]

    Returns:
        sample_abs (torch.tensor): shape [bs, 263, 1, 196]
    """
    n_joints = 22

    sample = dataset.t2m_dataset.inv_transform(sample_rel.cpu().permute(0, 2, 3, 1)).float()

    sample = recover_from_ric(sample, n_joints, abs_3d=False)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # [bs, 22, 3, 196]

    rot2xyz_pose_rep = 'xyz'
    rot2xyz_mask = None
    sample = model.rot2xyz(x=sample,
                           mask=rot2xyz_mask,
                           pose_rep=rot2xyz_pose_rep,
                           glob=True,
                           translation=True,
                           jointstype='smpl',
                           vertstrans=True,
                           betas=None,
                           beta=0,
                           glob_rot=None,
                           get_rotations_back=False)

    # sample now shape [32, 22, 3, 196].
    # from data_loaders.humanml.utils.plot_script import plot_3d_motion
    # plot_3d_motion("./test_positions_1.mp4", dataset.kinematic_chain, sample[4].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

    # Now convert skeleton back to sample with absolute representation
    sample_abs = dataset.motion_to_abs_data(sample, model)

    return sample_abs
