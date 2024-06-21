from torch.utils.data import DataLoader, Dataset
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import (CompMDMGeneratedDataset,
                                                                       CompMDMGeneratedDatasetCondition,
                                                                       CompMDMGeneratedDatasetInpainting)
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset_condmdi import CompMDMGeneratedDatasetCondMDI
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate
from utils.fixseed import fixseed
import os


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        trajs = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            traj = mm_motion['traj']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            traj = traj[None, :]
            # print("motion shape", motion.shape)
            # print("traj shape", traj.shape)
            motions.append(motion)
            trajs.append(traj)
            # import pdb; pdb.set_trace()
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        trajs = np.concatenate(trajs, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        trajs = trajs[sort_indx]
        return motions, m_lens, trajs


def get_motion_loader(opt_path, batch_size, ground_truth_dataset,
                      mm_num_samples, mm_num_repeats, device):
    opt = get_opt(opt_path, device, use_abs3d=...)
    raise NotImplementedError('This function is not used anymore. Use get_mdm_loader instead.')

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if 'v6' in opt.name:
        dataset = CompV6GeneratedDataset(opt, ground_truth_dataset,
                                         w_vectorizer, mm_num_samples,
                                         mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               drop_last=True,
                               num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader


# Our loader
def get_mdm_loader(model, diffusion, batch_size, ground_truth_loader,
                   mm_num_samples, mm_num_repeats, max_motion_length,
                   num_samples_limit, scale, seed, save_dir, full_inpaint):
    # set seed individually for each call
    fixseed(seed)
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    save_dir = os.path.join(save_dir, f'seed{seed:02d}')
    print('save_dir:', save_dir)
    if full_inpaint:
        dataset = CompMDMGeneratedDatasetInpainting(model, diffusion, ground_truth_loader,
                                                mm_num_samples, mm_num_repeats,
                                                max_motion_length, num_samples_limit,
                                                scale, save_dir=save_dir, seed=seed)
    else:
        dataset = CompMDMGeneratedDataset(model, diffusion, ground_truth_loader,
                                        mm_num_samples, mm_num_repeats,
                                        max_motion_length, num_samples_limit,
                                        scale, save_dir=save_dir, seed=seed)

    mm_dataset = MMGeneratedDataset(opt, dataset,
                                    ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               drop_last=True,
                               num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader


# Our loader for conditioning
def get_mdm_loader_cond(model_dict, diffusion_dict, batch_size, ground_truth_loader,
                   mm_num_samples, mm_num_repeats, max_motion_length,
                   num_samples_limit, scale, seed, save_dir, impute_until, skip_first_stage,
                   use_ddim):
    # set seed individually for each call
    fixseed(seed)
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    suffix = "_skip1st" if skip_first_stage else ""
    save_dir = os.path.join(save_dir, f't{impute_until:03d}{suffix}_seed{seed:02d}')
    print('save_dir:', save_dir)

    dataset = CompMDMGeneratedDatasetCondition(model_dict, diffusion_dict, ground_truth_loader,
                                      mm_num_samples, mm_num_repeats,
                                      max_motion_length, num_samples_limit,
                                      scale, save_dir=save_dir, impute_until=impute_until, skip_first_stage=skip_first_stage,
                                      seed=seed, use_ddim=use_ddim)

    mm_dataset = MMGeneratedDataset(opt, dataset,
                                    ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               drop_last=True,
                               num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader


def get_mdm_loader_ours(model_dict, diffusion_dict, batch_size, ground_truth_loader,
                   mm_num_samples, mm_num_repeats, max_motion_length,
                   num_samples_limit, text_scale, keyframe_scale, seed, save_dir, impute_until, skip_first_stage,
                   use_ddim, args):
    # set seed individually for each call
    fixseed(seed)
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    suffix = "_skip1st" if skip_first_stage else ""
    save_dir = os.path.join(save_dir, f't{impute_until:03d}{suffix}_seed{seed:02d}')
    print('save_dir:', save_dir)

    dataset = CompMDMGeneratedDatasetCondMDI(model_dict, diffusion_dict, ground_truth_loader,
                                      mm_num_samples, mm_num_repeats,
                                      max_motion_length, num_samples_limit,
                                      text_scale, keyframe_scale, save_dir=save_dir, impute_until=impute_until, skip_first_stage=skip_first_stage,
                                      seed=seed, use_ddim=use_ddim, args=args)

    mm_dataset = MMGeneratedDataset(opt, dataset,
                                    ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               drop_last=True,
                               num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader