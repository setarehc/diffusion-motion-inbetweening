from typing import Union

import torch
from torch import nn
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2, HumanML3D, TextOnlyDataset

from diffusion import gaussian_diffusion as gd
from diffusion.respace import DiffusionConfig, SpacedDiffusion, space_timesteps
from model.mdm import MDM
from model.mdm_dit import MDM_DiT
from model.mdm_unet import MDM_UNET
from utils.parser_util import DataOptions, DiffusionOptions, ModelOptions, TrainingOptions
from torch.utils.data import DataLoader

FullModelOptions = Union[DataOptions, ModelOptions, DiffusionOptions, TrainingOptions]
Datasets = Union[Text2MotionDatasetV2, HumanML3D, TextOnlyDataset]


def load_model_wo_clip(model: nn.Module, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                          strict=False)
    assert len(unexpected_keys) == 0, f'unexpected keys: {unexpected_keys}'
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args: FullModelOptions, data: DataLoader):
    if args.arch.startswith('dit'):
        # NOTE: adding 'two_head' in the args.arch would imply two_head=True
        # the model would predict both eps and x0.
        model = MDM_DiT(**get_model_args(args, data))
    elif args.arch.startswith('unet'):
        assert 'two_head' not in args.arch, 'unet does not support two_head'
        model = MDM_UNET(**get_model_args(args, data))
    else:
        model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args: FullModelOptions, data: DataLoader):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset == 'amass':
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        nfeats = 1
        if args.drop_redundant:
            njoints = 67 # 4 + 21 * 3
        else:
            njoints = 263
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == 'amass':
        data_rep = 'hml_vec' # FIXME: find what is the correct data rep
        njoints = 764
        nfeats = 1

    # Only produce trajectory (4 values: rot, x, z, y)
    if args.traj_only:
        njoints = 4
        nfeats = 1

    # whether to predict xstart and eps at the same time
    two_head = 'two_head' in args.arch

    return {
        'modeltype': '',
        'njoints': njoints,
        'nfeats': nfeats,
        'num_actions': num_actions,
        'translation': True,
        'pose_rep': 'rot6d',
        'glob': True,
        'glob_rot': True,
        'latent_dim': args.latent_dim,
        'ff_size': args.ff_size,
        'num_layers': args.layers,
        'num_heads': 4,
        'dropout': 0.1,
        'activation': "gelu",
        'data_rep': data_rep,
        'cond_mode': cond_mode,
        'cond_mask_prob': args.cond_mask_prob,
        'action_emb': action_emb,
        'arch': args.arch,
        'emb_trans_dec': args.emb_trans_dec,
        'clip_version': clip_version,
        'dataset': args.dataset,
        'two_head': two_head,
        'dim_mults': args.dim_mults,
        'adagn': args.unet_adagn,
        'zero': args.unet_zero,
        'unet_out_mult': args.out_mult,
        'tf_out_mult': args.out_mult,
        'xz_only': args.xz_only,
        'keyframe_conditioned': args.keyframe_conditioned,
        'keyframe_selection_scheme': args.keyframe_selection_scheme,
        'zero_keyframe_loss': args.zero_keyframe_loss,
    }


def create_gaussian_diffusion(args: FullModelOptions):
    steps = 1000  # 1000
    scale_beta = 1.  # no scaling
    if args.use_ddim:
        timestep_respacing = 'ddim100' # 'ddim100'  # can be used for ddim sampling, we don't use it.
    else:
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        conf=DiffusionConfig(
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON
                             if not args.predict_xstart else
                             gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE
                 if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=args.lambda_vel,
            lambda_rcxyz=args.lambda_rcxyz,
            lambda_fc=args.lambda_fc,
            clip_range=args.clip_range,
            train_trajectory_only_xz=args.xz_only,
            use_random_proj=args.use_random_proj,
            fp16=args.use_fp16,
            traj_only=args.traj_only,
            abs_3d=args.abs_3d,
            apply_zero_mask=args.apply_zero_mask,
            traj_extra_weight=args.traj_extra_weight,
            time_weighted_loss=args.time_weighted_loss,
            train_x0_as_eps=args.train_x0_as_eps,
        ),
    )


def load_saved_model(model, model_path, use_avg: bool=True):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model')
    load_model_wo_clip(model, state_dict)
    return model
