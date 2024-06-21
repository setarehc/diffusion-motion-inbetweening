# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from dataclasses import asdict
from functools import partial
from pprint import pprint
from utils.fixseed import fixseed
import os
import time
import numpy as np
import torch
import copy
import json
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.output_util import sample_to_motion, construct_template_variables, save_multiple_samples
from utils.generation_template import get_template
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders import humanml_utils
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
# import flag
from torch.cuda import amp
from sample.gmd.condition import (get_target_from_kframes, get_inpainting_motion_from_traj,
                              get_target_and_inpt_from_kframes_batch,
                              cond_fn_key_location, cond_fn_sdf, log_trajectory_from_xstart,
                              CondKeyLocations, CondKeyLocationsWithSdf)

from sample.gmd.keyframe_pattern import get_kframes, get_obstacles
# For debugging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from utils.editing_util import load_fixed_dataset, get_keyframes_mask


def load_reward_model(data):
    '''
    Create a reward model to help computing grad_{x_t} for traj conditioning.
    '''
    args_reward = generate_args(trajectory_model=True)  #
    args_reward.model_path = "./save/my_traj/model000400000.pt"
    args_reward.predict_xstart = True
    args_reward.abs_3d = True
    args_reward.traj_only = True

    reward_model, _ = create_model_and_diffusion(args_reward, data)
    print(
        f"Loading reward model checkpoints from [{args_reward.model_path}]..."
    )
    load_saved_model(reward_model, args_reward.model_path)  # , use_avg_model=args_reward.gen_avg_model)

    if args_reward.guidance_param != 1:
        reward_model = ClassifierFreeSampleModel(
            reward_model
        )  # wrapping model with the classifier-free sampler
    reward_model.to(dist_util.dev())
    reward_model.eval()  # disable random masking
    return reward_model


def load_traj_model(data):
    '''
    The trajectory model predicts trajectory that will be use for infilling in motion model.
    Create a trajectory model that produces trajectory to be inptained by the motion model.
    '''
    print("Setting traj model ...")
    # NOTE: Hard-coded trajectory model location
    traj_model_path = "./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt"
    args_traj = generate_args(model_path=traj_model_path)

    # print(args_traj.__dict__)
    # print(args_traj.arch)
    traj_model, traj_diffusion = create_model_and_diffusion(args_traj, data)

    print(f"Loading traj model checkpoints from [{args_traj.model_path}]...")
    load_saved_model(traj_model, args_traj.model_path)

    if args_traj.guidance_param != 1:
        traj_model = ClassifierFreeSampleModel(
            traj_model)  # wrapping model with the classifier-free sampler
    traj_model.to(dist_util.dev())
    traj_model.eval()  # disable random masking
    return traj_model, traj_diffusion


def main():
    args = generate_args()
    print(args.__dict__)
    print(args.arch)
    print("##### Additional Guidance Mode: %s #####" % args.guidance_mode)

    # Update args according to guidance mode
    # args = get_template(args, template_name="testing")
    # args = get_template(args, template_name="kps") # mdm_legacy # trajectory # guidance_mode
    args = get_template(args, template_name=args.guidance_mode)

    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    cut_frames = int(args.motion_length_cut * fps)
    print('n_frames', n_frames)
    is_using_data = not any([
        args.input_text, args.text_prompt, args.action_file, args.action_name
    ])
    dist_util.setup_dist(args.device)
    # Output directory
    if out_path == '':
        # out_path = os.path.join(os.path.dirname(args.model_path),
        #                         'samples_{}_{}_seed{}_{}'.format(name, niter, args.seed, time.strftime("%Y%m%d-%H%M%S")))
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_seed{}'.format(niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace(
                '.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace(
                '.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
        # Do 3 repetitions from the same propmt. But put it in num_sample instead so we can do all of them in parallel
        # args.num_samples = 3
        args.num_repetitions = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    # NOTE: Currently not supporting multiple repetitions due to the way we handle trajectory model
    args.num_repetitions = 1

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################
    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{
            'inp': torch.zeros(n_frames),
            'tokens': None,
            # this would be incorrect for UNET models
            # 'lengths': n_frames,
            'lengths': cut_frames,
        }] * args.num_samples
        # model_kwargs['y']['lengths']
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [
                dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
            ]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [
                dict(arg, action=one_action, action_text=one_action_text)
                for arg, one_action, one_action_text in zip(
                    collate_args, action, action_text)
            ]

        _, model_kwargs = collate(collate_args)

    # Name for logging
    model_kwargs['y']['log_name'] = out_path
    # import pdb; pdb.set_trace()
    model_kwargs['y']['traj_model'] = args.traj_only
    # TODO: move two-staged model to a new class
    #########################################
    # Load another model for reward function
    if args.gen_reward_model:
        reward_model = load_reward_model(data)
        reward_model_kwargs = copy.deepcopy(model_kwargs)
    #########################################
    # loading another model for trajectory conditioning
    if args.gen_two_stages:
        traj_model, traj_diffusion = load_traj_model(data)
        traj_model_kwargs = copy.deepcopy(model_kwargs)
        traj_model_kwargs['y']['log_name'] = out_path
        traj_model_kwargs['y']['traj_model'] = True
    #############################################

    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []

    # NOTE: test for classifier-free sampling
    USE_CLASSIFIER_FREE = False # True

    model_device = next(model.parameters()).device
    # Load preprocessed file for inpainting test
    # [3, 263, 1, 120]
    input_motions, ground_positions = load_processed_file(model_device, args.batch_size, args.traj_only)

    if args.use_fixed_dataset: # TODO: this is for debugging - need a neater way to do this for the final version - num_samples should be 10
        import pdb; pdb.set_trace()
        assert args.dataset == 'humanml' and args.abs_3d
        input_motions, model_kwargs_new = load_fixed_dataset(args.num_samples)
        for key in model_kwargs_new['y'].keys():
            assert key in model_kwargs['y']
            model_kwargs['y'][key] = model_kwargs_new['y'][key]

    if not args.use_fixed_dataset:
        input_skels = recover_from_ric(input_motions.permute(0, 2, 3, 1), 22, abs_3d=False)
        input_skels = input_skels.squeeze(1)
        # input_skels = input_skels[0].transpose(0, 3, 1, 2)
        # Get key frames for guidance
        if args.guidance_mode == "trajectory" or args.guidance_mode == "mdm_legacy":
            # Get key frames for guidance
            kframes = get_kframes(ground_positions) # ground_positions=ground_positions)
            # model_kwargs['y']['kframes_pattern'] = kframes
        elif args.guidance_mode == "kps":
            if args.interactive:
                # Get key frames for guidance from interactive GUI
                kframes = ()
            else:
                kframes = get_kframes(pattern="zigzag")
            model_kwargs['y']['kframes_pattern'] = kframes
        elif args.guidance_mode == "sdf":
            kframes = get_kframes(pattern="sdf")
            model_kwargs['y']['kframes_pattern'] = kframes
            obs_list = get_obstacles()
        elif USE_CLASSIFIER_FREE:
            kframes = get_kframes(pattern="zigzag", interpolate=True)

            # kframes = []
            # kframes = get_kframes(ground_positions) # ground_positions=ground_positions)
            model_kwargs['y']['kframes_pattern'] = kframes
        else:
            kframes = []
    else:
        keyframes_mask, obs_joint_mask = get_keyframes_mask(data=input_motions, lengths=model_kwargs["y"]["lengths"], edit_mode='benchmark_sparse',
                                                                  feature_mode='pos', trans_length=20,
                                                                  get_joint_mask=True) # [nsamples, njoints, nfeats, nframes]
        kframes = []
        for i, idx in enumerate(torch.where(keyframes_mask)[0]):
            kframes.append(tuple((idx, tuple(input_motions[0,[1,2],0,idx].tolist()))))

    if args.guidance_mode == "mdm_legacy" and args.do_inpaint:
        # When use MDM (or relative model) and inpainting, be sure to set imputation mode
        # to "IMPUTE_AT_X0 = False" in gaussian_diffusion.py
        model_kwargs['y']['impute_relative'] = True
        model_kwargs['y']['inpainted_motion'] = input_motions.clone()
        # Get impainting mask
        inpainting_mask = torch.tensor(
            humanml_utils.HML_ROOT_MASK,
            dtype=torch.bool,
            device=model_device)  # True is root (global location)
        # Do not need to fix y
        inpainting_mask[3:] = False
        # model_kwargs['y']['inpainting_mask'][0] = False
        inpainting_mask = inpainting_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
                input_motions.shape[0], 1, input_motions.shape[2],
                input_motions.shape[3])
        model_kwargs['y']['inpainting_mask'] = inpainting_mask

        motion_cond_until = 0
        motion_impute_until = 0
    else:
        motion_cond_until = 20
        motion_impute_until = 1

    #### Standardized conditioning
    # TODO: clean this for each guidance mode
    # Use the same function call as used during evaluation (condition.py)
    kframes_num = [a for (a,b) in kframes] # [0, 30, 60, 90, 119] ***
    kframes_posi = torch.tensor(kframes_num, dtype=torch.int).unsqueeze(0).repeat(args.batch_size, 1)

    ### Prepare target
    # Get dummy skel_motions of shape [1, 22, 3, max_length] from keyframes
    # We do it this way so that get_target_...() can be shared across guidance modes.
    dummy_skel_motions = torch.zeros([1, 22, 3, n_frames])
    for (tt, locs) in kframes:
        print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
        dummy_skel_motions[0, 0, [0, 2], tt] = torch.tensor([locs[0], locs[1]])
    dummy_skel_motions = dummy_skel_motions.repeat(args.batch_size, 1, 1, 1)  # [1, 22, 3, max_length]

    (target, target_mask,
        inpaint_traj_p2p, inpaint_traj_mask_p2p,
        inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion_p2p, inpaint_mask_p2p,
        inpaint_motion_points, inpaint_mask_points) = get_target_and_inpt_from_kframes_batch(dummy_skel_motions, kframes_posi, data.dataset)
    target = target.to(model_device)
    target_mask = target_mask.to(model_device)
    model_kwargs['y']['target'] = target
    model_kwargs['y']['target_mask'] = target_mask
    ###########################################
    # NOTE: Test imputing with poses
    # TODO: Delete this
    GUIDE_WITH_POSES = False
    # GUIDE_WITH_POSES = True
    if GUIDE_WITH_POSES:
        print("Guide with poses")
        # Get impainting mask
        inpainting_mask = torch.tensor(
            np.array([False] + [True]*2 + [False] + [True]*63 + [False]*(263-67)),
            dtype=torch.bool,
            device=model_device)  # True is root (global location)
        # model_kwargs['y']['inpainting_mask'][0] = False
        inpainting_mask = inpainting_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
                input_motions.shape[0], 1, input_motions.shape[2],
                n_frames)
        model_kwargs['y']['inpainting_mask'] = inpainting_mask
        inpaint_mask_points = inpainting_mask
        input_motions[:, [1, 2], 0, :] = torch.from_numpy(ground_positions[:, 0, [0, 2]]).to(input_motions.device).permute(1, 0).unsqueeze(0).repeat(3, 1, 1)
        inpaint_motion_points = torch.cat([input_motions,
                                        torch.zeros(*input_motions.shape[:3], n_frames - input_motions.shape[3], device=input_motions.device)], dim=3)
        for i in range(inpaint_mask_points.shape[-1]):
            if i not in [kk for (kk, _) in kframes]:
                inpaint_mask_points[:, :, :, i] = False
    ###########################################

    # Output path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    ############################################
    # Generate trajectory
    # NOTE: num_repetitions > 1 is currently not supported
    for rep_i in range(args.num_repetitions):
        assert args.num_repetitions == 1, "Not implemented"

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['text_scale'] = torch.ones(
                args.batch_size, device=dist_util.dev()) * args.guidance_param
            if args.gen_reward_model:
                reward_model_kwargs['y']['text_scale'] = torch.ones(
                    args.batch_size,
                    device=dist_util.dev()) * args.guidance_param
            if args.gen_two_stages:
                traj_model_kwargs['y']['text_scale'] = torch.ones(
                    args.batch_size,
                    device=dist_util.dev()) * args.guidance_param

        print("classifier scale", args.classifier_scale)

        # Standardized conditioning
        impute_slack = 20
        impute_until = 100

        #####################################################
        # If using TWO_STAGES, generate the trajectory first
        if args.gen_two_stages:
            traj_model_kwargs['y']['log_id'] = rep_i
            ### Standardized conditioning
            if args.p2p_impute:
                ### Inpaint with p2p
                traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p.to(model_device)
                traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p.to(model_device)
            else:
                ### Inpaint with kps
                traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj_points.to(model_device)
                traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_points.to(model_device)

            # Set when to stop imputing
            traj_model_kwargs['y']['cond_until'] = impute_slack
            traj_model_kwargs['y']['impute_until'] = impute_until
            # NOTE: We have the option of switching the target motion from line to just key locations
            # We call this a 'second stage', which will start after t reach 'impute_until'
            traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
            traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
            traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)

            traj_diffusion.data_transform_fn = None
            traj_diffusion.data_inv_transform_fn = None
            traj_diffusion.log_trajectory_fn = partial(
                log_trajectory_from_xstart,
                kframes=kframes,
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                abs_3d=args.abs_3d,
                traject_only=True,
                n_frames=cut_frames,
                combine_to_video=True,
                obs_list=obs_list)

            if args.guidance_mode == "kps":
                cond_fn_traj = CondKeyLocations(target=target,
                                            target_mask=target_mask,
                                            transform=data.dataset.t2m_dataset.transform_th,
                                            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                            abs_3d=args.abs_3d,
                                            classifiler_scale=args.classifier_scale,
                                            use_mse_loss=args.gen_mse_loss,
                                            use_rand_projection=False,
                                            )
                traj_model_kwargs['y']['gmd'] = True
            elif args.guidance_mode == "sdf":
                cond_fn_traj = CondKeyLocationsWithSdf(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=False,
                                        obs_list=obs_list
                                        )
            else:
                cond_fn_traj = None

            sample_fn = traj_diffusion.p_sample_loop
            dump_steps = [1, 100, 300, 500, 700, 850, 999]

            l = []
            import time
            assert args.batch_size == 1
            for _ in range(10):
                start = time.time()
                traj_sample = sample_fn(
                    traj_model,
                    (args.batch_size, traj_model.njoints, traj_model.nfeats,
                    n_frames),
                    clip_denoised=True,  # False,
                    model_kwargs=traj_model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None, # None,
                    progress=True,
                    dump_steps=dump_steps,  # None,
                    noise=None,
                    const_noise=False,
                    cond_fn=cond_fn_traj,
                )
                l.append(time.time() - start)
            print(f'***DONE****')

            # Set inpainting information for motion model
            traj_motion, traj_mask = get_inpainting_motion_from_traj(
                traj_sample[-1],
                inv_transform_fn=data.dataset.t2m_dataset.inv_transform_th,
            )
            # plt.scatter(traj_motion[0, 1, 0, :120].cpu().numpy(), traj_motion[0, 2, 0, :120].cpu().numpy())
            model_kwargs['y']['inpainted_motion'] = traj_motion
            model_kwargs['y']['inpainting_mask'] = traj_mask
            # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
            target = torch.zeros([args.batch_size, n_frames, 22, 3], device=traj_motion.device)
            target_mask = torch.zeros_like(target, dtype=torch.bool)
            # This assume that the traj_motion is in the 3D space without normalization
            # traj_motion: [3, 263, 1, 196]
            target[:, :, 0, [0, 2]] = traj_motion.permute(0, 3, 2, 1)[:, :, 0,[1, 2]]
            # target_mask[:, :int(flag.GEN_MOTION_LENGTH_CUT * 20.0), 0, [0, 2]] = True
            target_mask[:, :, 0, [0, 2]] = True

        elif not args.guidance_mode == "mdm_legacy":
            model_kwargs['y']['inpainted_motion'] = inpaint_motion_points.to(model_device) # inpaint_motion_p2p
            model_kwargs['y']['inpainting_mask'] = inpaint_mask_points.to(model_device)  # inpaint_p2p_mask

        if args.use_ddim:
            sample_fn = diffusion.ddim_sample_loop
            # dump_steps for logging progress
            dump_steps = [1, 10, 30, 50, 70, 85, 99]
        else:
            sample_fn = diffusion.p_sample_loop
            # dump_steps = [1, 100, 300, 500, 700, 850, 999]
            dump_steps = [999]

        # NOTE: Delete inpainting information if not using it. Just to be safe
        # TODO: remove this
        if not args.do_inpaint and "inpainted_motion" in model_kwargs['y'].keys():
            del model_kwargs['y']['inpainted_motion']
            del model_kwargs['y']['inpainting_mask']

        # Name for logging
        model_kwargs['y']['log_id'] = rep_i
        model_kwargs['y']['cond_until'] = motion_cond_until  # impute_slack
        model_kwargs['y']['impute_until'] = motion_impute_until # 20  # impute_slack
        # Pass functions to the diffusion
        diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
        diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
        diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th
        diffusion.log_trajectory_fn = partial(
            log_trajectory_from_xstart,
            kframes=kframes,
            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
            abs_3d=args.abs_3d,
            use_rand_proj=args.use_random_proj,
            traject_only=args.traj_only,
            n_frames=cut_frames,
            combine_to_video=True,
            obs_list=obs_list)

        # diffusion.log_trajectory_fn(input_motions.detach(), out_path, [1000], torch.tensor([1000] * args.batch_size), model_kwargs['y']['log_id'])
        # diffusion.log_trajectory_fn(model_kwargs['y']['inpainted_motion'].detach(), out_path, [1000], torch.tensor([1000] * args.batch_size), model_kwargs['y']['log_id'])

        # TODO: move the followings to a separate function
        if args.guidance_mode == "kps" or args.guidance_mode == "trajectory":
            cond_fn = CondKeyLocations(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=args.use_random_proj
                                        )
            model_kwargs['y']['gmd'] = True
            model_kwargs['y']['traj_model'] = False
            # import pdb; pdb.set_trace()
        elif args.guidance_mode == "sdf":
            cond_fn = CondKeyLocationsWithSdf(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=args.use_random_proj,
                                        obs_list=obs_list
                                        )
        elif args.guidance_mode == "no" or args.guidance_mode == "mdm_legacy":
            cond_fn = None

        ###################
        # MODEL INFERENCING
        # list of [bs, njoints, nfeats, nframes] each element is a different time step
        for idx in range(10):
            start = time.time()
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                # clip_denoised=False,
                clip_denoised=not args.predict_xstart,

                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
                init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
                progress=True,
                dump_steps=dump_steps,  # None,
                noise=None,
                const_noise=False,
                cond_fn=cond_fn,
            )
            l[idx] = l[idx] +(time.time() - start)

        print(f'Sampling took {np.mean(l)} +- {np.std(l)} seconds')

        # Cut the generation to the desired length
        # NOTE: this is important for UNETs where the input must be specific size (e.g. 224)
        # but the output can be cut to any length
        gen_eff_len = min(sample[0].shape[-1], cut_frames)
        print('cut the motion length to', gen_eff_len)
        for j in range(len(sample)):
            sample[j] = sample[j][:, :, :, :gen_eff_len]
        ###################

        num_dump_step = len(dump_steps)
        args.num_dump_step = num_dump_step
        # Convert sample to XYZ skeleton locations
        # Each return size [bs, 1, 3, 120]
        cur_motions, cur_lengths, cur_texts = sample_to_motion(
            sample, args, model_kwargs, model, gen_eff_len,
            data.dataset.t2m_dataset.inv_transform)
        all_motions.extend(cur_motions)
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)

    ### Save videos
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions,
                                 axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[:total_num_samples]  # #       not sure? [bs, njoints, 6, seqlen]
    all_text = all_text[:
                        total_num_samples]  # len() = args.num_samples * num_dump_step
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path, {
            'motion': all_motions,
            'text': all_text,
            'lengths': all_lengths,
            'num_samples': args.num_samples,
            'num_repetitions': args.num_repetitions
        })
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    if args.traj_only:
        skeleton = [[0, 0]]

    sample_files = []
    num_samples_in_out_file = num_dump_step  # 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    # NOTE: we change the behavior of num_samples to support visualising denoising progress with multiple dump steps
    # for sample_i in range(args.num_samples * num_dump_step): # range(args.num_samples):
    # for sample_i in range(args.num_repetitions): # range(args.num_samples):
    for sample_i in range(args.num_samples):
        rep_files = []
        # for rep_i in range(args.num_repetitions):
        # for rep_i in range(num_dump_step):
        for dump_step_i in range(num_dump_step):
            # idx = rep_i + sample_i * num_dump_step # rep_i*args.batch_size + sample_i
            # idx = sample_i * num_dump_step + dump_step_i
            idx = sample_i + dump_step_i * args.num_samples
            print("saving", idx)
            caption = all_text[idx]
            length = all_lengths[idx]
            motion = all_motions[idx].transpose(2, 0,
                                                1)[:length]  # [120, 22, 3]
            save_file = sample_file_template.format(sample_i, dump_step_i)
            print(
                sample_print_template.format(caption, sample_i, dump_step_i,
                                             save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path,
                           skeleton,
                           motion,
                           dataset=args.dataset,
                           title=caption,
                           fps=fps,
                           traj_only=args.traj_only,
                           kframes=kframes,
                           obs_list=obs_list,
                           # NOTE: TEST
                           target_pose=input_skels[0].cpu().numpy(),
                           gt_frames=[kk for (kk, _) in kframes] if GUIDE_WITH_POSES else [])
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            args, out_path, row_print_template, all_print_template,
            row_file_template, all_file_template, caption,
            num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, max_frames, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='test',
        hml_mode='text_only', # 'train'
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    # what's this for?
    data.fixed_length = n_frames
    return data


def load_processed_file(model_device, batch_size, traject_only=False):
    '''Load template file for trajectory imputing'''
    template_path = "./assets/template_joints.npy"
    init_joints = torch.from_numpy(np.load(template_path))
    from data_loaders.humanml.scripts.motion_process import process_file, recover_root_rot_pos
    data, ground_positions, positions, l_velocity = process_file(
        init_joints.permute(0, 3, 1, 2)[0], 0.002)
    init_image = data
    # make it (1, 263, 1, 120)
    init_image = torch.from_numpy(init_image).unsqueeze(0).float()
    init_image = torch.cat([init_image, init_image[0:1, 118:119, :].clone()],
                           dim=1)
    # Use transform_fn instead
    # init_image = (init_image - data.dataset.t2m_dataset.mean) / data.dataset.t2m_dataset.std
    init_image = init_image.unsqueeze(1).permute(0, 3, 1, 2)
    init_image = init_image.to(model_device)
    if traject_only:
        init_image = init_image[:, :4, :, :]

    init_image = init_image.repeat(batch_size, 1, 1, 1)
    return init_image, ground_positions


if __name__ == "__main__":
    main()
