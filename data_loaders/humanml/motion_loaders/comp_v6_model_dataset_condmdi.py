import torch
from utils.fixseed import fixseed
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
import os
import copy
from functools import partial

from data_loaders.humanml.data.dataset import abs3d_to_rel, sample_to_motion, rel_to_abs3d
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.metrics import calculate_skating_ratio
from sample.gmd.condition import (cond_fn_key_location, get_target_from_kframes, get_target_and_inpt_from_kframes_batch,
                              log_trajectory_from_xstart, get_inpainting_motion_from_traj, get_inpainting_motion_from_gt,
                              cond_fn_key_location, compute_kps_error, cond_fn_sdf,
                              CondKeyLocations, CondKeyLocationsWithSdf, compute_kps_error_arbitrary)
from utils.editing_util import get_keyframes_mask


# Data class for generated motion by *conditioning*
class CompMDMGeneratedDatasetCondMDI(Dataset):

    def __init__(self, model_dict, diffusion_dict, dataloader, mm_num_samples, mm_num_repeats,
                 max_motion_length, num_samples_limit, text_scale=1., keyframe_scale=1., save_dir=None, impute_until=0, skip_first_stage=False,
                 seed=None, use_ddim=False, args=None):

        assert seed is not None, "must provide seed"
        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.save_dir = save_dir
        # This affect the trajectory model if we do two-stage, if not, it will affect the motion model
        # For trajectory model, the output traj will be imptued until 20 (set by impute_slack)
        self.impute_until = impute_until

        motion_model, traj_model = model_dict["motion"], model_dict["traj"]
        motion_diffusion, traj_diffusion = diffusion_dict["motion"], diffusion_dict["traj"]

        ### Basic settings
        # motion_classifier_scale = 100.0
        # print("motion classifier scale", motion_classifier_scale)
        log_motion = False
        # guidance_mode = 'no'
        abs_3d = True
        use_random_proj = self.dataset.use_rand_proj
        # print("guidance mode", guidance_mode)
        print("use ddim", use_ddim)

        model_device = next(motion_model.parameters()).device
        motion_diffusion.data_get_mean_fn = self.dataset.t2m_dataset.get_std_mean
        motion_diffusion.data_transform_fn = self.dataset.t2m_dataset.transform_th
        motion_diffusion.data_inv_transform_fn = self.dataset.t2m_dataset.inv_transform_th
        if log_motion:
            motion_diffusion.log_trajectory_fn = partial(
                log_trajectory_from_xstart,
                kframes=[],
                inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                abs_3d=abs_3d,  # <--- assume the motion model is absolute
                use_rand_proj=self.dataset.use_rand_proj,
                traject_only=False,
                n_frames=max_motion_length)

        assert save_dir is not None
        assert mm_num_samples < len(dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        # use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        # clip_denoised = True
        self.max_motion_length = max_motion_length

        # sample_fn_motion = (
        #     motion_diffusion.p_sample_loop if not use_ddim else motion_diffusion.ddim_sample_loop
        # )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        # NOTE: mm = multi-modal
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        motion_model.eval()

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                '''For each datapoint, we do the following
                    1. Sample 3-10 (?) points from the ground truth trajectory to be used as conditions
                    2. Generate trajectory with trajectory model
                    3. Generate motion based on the generated traj using inpainting and cond_fn.
                '''

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                # add CFG scale to batch
                if args.guidance_param != 1:
                    # text classifier-free guidance
                    model_kwargs['y']['text_scale'] = torch.ones(motion.shape[0], device=dist_util.dev()) * text_scale
                if args.keyframe_guidance_param != 1:
                    # keyframe classifier-free guidance
                    model_kwargs['y']['keyframe_scale'] = torch.ones(motion.shape[0], device=dist_util.dev()) * keyframe_scale

                ### 1. Prepare motion for conditioning ###
                model_kwargs['y']['traj_model'] = False

                # Convert to 3D motion space
                # NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_rand_proj' is True
                # NOTE: the 'motion' will have relative representation if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_abs3d' is True
                gt_poses = motion.permute(0, 2, 3, 1)
                gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263] # TODO: mean and std are absolute mean and std and this is done on purpose! Why?  dataset: The 'eval' is here because we want inv_transform to work the same way at inference for model with abs3d,regradless of which dataset is loaded.
                # TODO: gt_poses = gt_poses * self.dataset.std_rel + self.dataset.mean_rel
                # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=False)
                gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                                                    jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # gt_skel_motions shape [32, 22, 3, 196]
                # Visualize to make sure it is correct
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # plot_3d_motion("./gt_source_abs.mp4", self.dataset.kinematic_chain,
                #                gt_skel_motions[0].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)


                ### START TEST ###
                # gt_poses = motion.permute(0, 2, 3, 1)
                # gt_poses = gt_poses * self.dataset.std_rel + self.dataset.mean_rel
                # # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                # gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=False)
                # gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                # gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                #                                     jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # # gt_skel_motions shape [32, 22, 3, 196]
                # # Visualize to make sure it is correct
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # plot_3d_motion("./gt_source_rel.mp4", self.dataset.kinematic_chain,
                #                gt_skel_motions[0].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
                # Sample gt_source_abs.mp4 looks better
                ### END TEST ###

                # Convert relative representation to absolute representation for ground-truth motions
                motion_abs = rel_to_abs3d(sample_rel=motion, dataset=self.dataset, model=motion_model).to(dist_util.dev())
                ### START TEST ###
                # Visualize to make sure it is correct
                # gt_poses = model_kwargs['y']['inpainted_motion'].permute(0, 2, 3, 1)
                # gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263]
                # # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                # gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=True)
                # gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                # gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                #                                     jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # plot_3d_motion("./test_rel2glob_gt.mp4", self.dataset.kinematic_chain,
                #                gt_skel_motions[0].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
                # Sample matches gt_source_abs.mp4
                ### END TEST ###

                # import pdb; pdb.set_trace()

                ### START OUR BUILD OF model_kwargs ###
                if motion_model.keyframe_conditioned:
                    # Conditional synthesis arguments:
                    keyframes_indices, joint_mask = self.set_conditional_synthesis_args(model_kwargs, motion_abs)
                elif self.args.imputate or self.args.reconstruction_guidance:
                    # Editing arguments:
                    keyframes_indices, joint_mask = self.set_inference_editing_args(model_kwargs, motion_abs)
                ### END OUR BUILD OF model_kwargs ###

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                mm_trajectories = []
                for t in range(repeat_times):
                    seed_number = seed * 100_000 + i * 100 + t
                    fixseed(seed_number)
                    batch_file = f'{i:04d}_{t:02d}.pt'
                    batch_path = os.path.join(self.save_dir, batch_file)

                    # reusing the batch if it exists
                    # if os.path.exists(batch_path):
                    if False:  # GUY - IGNORE CACHE FOR NOW
                        # [bs, njoints, nfeat, seqlen]
                        sample_motion = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:
                        print(f'working on {batch_file}')
                        # for smoother motions
                        # impute_slack = 20
                        # NOTE: For debugging
                        # traj_model_kwargs['y']['log_name'] = self.save_dir
                        # traj_model_kwargs['y']['log_id'] = i
                        model_kwargs['y']['log_name'] = self.save_dir
                        model_kwargs['y']['log_id'] = i
                        # motion model always impute until 20
                        # model_kwargs['y']['cond_until'] = impute_slack
                        # model_kwargs['y']['impute_until'] = impute_slack

                        # if skip_first_stage:
                        #     # No first stage. Skip straight to second stage
                        #     ### Add motion to inpaint
                        #     # import pdb; pdb.set_trace()
                        #     # del model_kwargs['y']['inpainted_motion']
                        #     # del model_kwargs['y']['inpainting_mask']
                        #     model_kwargs['y']['inpainted_motion'] = inpaint_motion.to(model_device) # init_motion.to(model_device)
                        #     model_kwargs['y']['inpainting_mask'] = inpaint_mask.to(model_device)
                        #
                        #     model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_motion_points.to(model_device)
                        #     model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_mask_points.to(model_device)
                        #     # import pdb; pdb.set_trace()
                        #
                        #     # For classifier-free
                        #     CLASSIFIER_FREE = True
                        #     if CLASSIFIER_FREE:
                        #         impute_until = 1
                        #         impute_slack = 20
                        #         # del model_kwargs['y']['inpainted_motion']
                        #         # del model_kwargs['y']['inpainting_mask']
                        #         model_kwargs['y']['inpainted_motion'] = inpaint_motion_points.to(model_device) # init_motion.to(model_device)
                        #         model_kwargs['y']['inpainting_mask'] = inpaint_mask_points.to(model_device)
                        #
                        #     # Set when to stop imputing
                        #     model_kwargs['y']['cond_until'] = impute_slack
                        #     model_kwargs['y']['impute_until'] = impute_until
                        #     model_kwargs['y']['impute_until_second_stage'] = impute_slack
                        #
                        # else:
                        #     ### Add motion to inpaint
                        #     traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj.to(model_device) # init_motion.to(model_device)
                        #     traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask.to(model_device)
                        #
                        #     # Set when to stop imputing
                        #     traj_model_kwargs['y']['cond_until'] = impute_slack
                        #     traj_model_kwargs['y']['impute_until'] = impute_until
                        #     # NOTE: We have the option of switching the target motion from line to just key locations
                        #     # We call this a 'second stage', which will start after t reach 'impute_until'
                        #     traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
                        #     traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
                        #     traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)
                        #
                        #
                        #     ##########################################################
                        #     # print("************* Test: not using dense gradient ****************")
                        #     # NO_GRAD = True
                        #     # traj_model_kwargs['y']['cond_until'] = 1000
                        #
                        #     # traj_model_kwargs['y']['impute_until'] = 1000
                        #     # traj_model_kwargs['y']['impute_until_second_stage'] = 0
                        #
                        #     ##########################################################
                        #
                        #     ### Generate trajectory
                        #     # [bs, njoints, nfeat, seqlen]
                        #     # NOTE: add cond_fn
                        #     sample_traj = sample_fn_traj(
                        #         traj_model,
                        #         inpaint_traj.shape,
                        #         clip_denoised=clip_denoised,
                        #         model_kwargs=traj_model_kwargs,  # <-- traj_kwards
                        #         skip_timesteps=0,  # NOTE: for debugging, start from 900
                        #         init_image=None,
                        #         progress=True,
                        #         dump_steps=None,
                        #         noise=None,
                        #         const_noise=False,
                        #         cond_fn=partial(
                        #             cond_fn_key_location, # cond_fn_sdf, #,
                        #             transform=self.dataset.t2m_dataset.transform_th,
                        #             inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                        #             target=target,
                        #             target_mask=target_mask,
                        #             kframes=[],
                        #             abs_3d=abs_3d, # <<-- hard code,
                        #             classifiler_scale=trajectory_classifier_scale,
                        #             use_mse_loss=False),  # <<-- hard code
                        #     )
                        #
                        #     ### Prepare conditions for motion from generated trajectory ###
                        #     # Get inpainting information for motion model
                        #     traj_motion, traj_mask = get_inpainting_motion_from_traj(
                        #         sample_traj, inv_transform_fn=self.dataset.t2m_dataset.inv_transform_th)
                        #     # Get target for loss grad
                        #     # Target has dimention [bs, max_motion_length, 22, 3]
                        #     target = torch.zeros([motion.shape[0], max_motion_length, 22, 3], device=traj_motion.device)
                        #     target_mask = torch.zeros_like(target, dtype=torch.bool)
                        #     # This assume that the traj_motion is in the 3D space without normalization
                        #     # traj_motion: [3, 263, 1, 196]
                        #     target[:, :, 0, [0, 2]] = traj_motion.permute(0, 3, 2, 1)[:, :, 0,[1, 2]]
                        #     target_mask[:, :, 0, [0, 2]] = True
                        #     # Set imputing trajectory
                        #     model_kwargs['y']['inpainted_motion'] = traj_motion
                        #     model_kwargs['y']['inpainting_mask'] = traj_mask
                        #     ### End - Prepare conditions ###

                        ### Generate motion
                        # NOTE: add cond_fn
                        # TODO: move the followings to a separate function
                        # if guidance_mode == "kps" or guidance_mode == "trajectory":
                        #     cond_fn = CondKeyLocations(target=target,
                        #                                 target_mask=target_mask,
                        #                                 transform=self.dataset.t2m_dataset.transform_th,
                        #                                 inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                        #                                 abs_3d=abs_3d,
                        #                                 classifiler_scale=motion_classifier_scale,
                        #                                 use_mse_loss=False,
                        #                                 use_rand_projection=self.dataset.use_random_proj
                        #                                 )
                        # # elif guidance_mode == "sdf":
                        # #     cond_fn = CondKeyLocationsWithSdf(target=target,
                        # #                                 target_mask=target_mask,
                        # #                                 transform=data.dataset.t2m_dataset.transform_th,
                        # #                                 inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                        # #                                 abs_3d=abs_3d,
                        # #                                 classifiler_scale=motion_classifier_scale,
                        # #                                 use_mse_loss=False,
                        # #                                 use_rand_projection=self.dataset.use_random_proj,
                        # #                                 obs_list=obs_list
                        # #                                 )
                        # elif guidance_mode == "no" or guidance_mode == "mdm_legacy":
                        # cond_fn = None

                        # if NO_GRAD:
                        #     cond_fn = None
                        sample_fn = motion_diffusion.p_sample_loop

                        sample_motion = sample_fn(
                            motion_model,
                            (motion.shape[0], motion_model.njoints, motion_model.nfeats, motion.shape[3]),
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,  # True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                        )

                        # sample_motion = sample_fn_motion(
                        #     motion_model,
                        #     (motion.shape[0], motion_model.njoints, motion_model.nfeats, motion.shape[3]),  # motion.shape
                        #     clip_denoised=clip_denoised,
                        #     model_kwargs=model_kwargs,
                        #     skip_timesteps=0,
                        #     init_image=None,
                        #     progress=True,
                        #     dump_steps=None,
                        #     noise=None,
                        #     const_noise=False,
                        #     cond_fn=cond_fn
                        #         # partial(
                        #         # cond_fn_key_location,
                        #         # transform=self.dataset.t2m_dataset.transform_th,
                        #         # inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                        #         # target=target,
                        #         # target_mask=target_mask,
                        #         # kframes=[],
                        #         # abs_3d=True, # <<-- hard code,
                        #         # classifiler_scale=motion_classifier_scale,
                        #         # use_mse_loss=False),  # <<-- hard code
                        # )
                        # save to file
                        torch.save(sample_motion, batch_path)


                    # print('cut the motion length from {} to {}'.format(sample_motion.shape[-1], self.max_motion_length))
                    sample = sample_motion[:, :, :, :self.max_motion_length]

                    # Compute error for key xz locations
                    cur_motion = sample_to_motion(sample, self.dataset, motion_model)
                    #kps_error = compute_kps_error(cur_motion, gt_skel_motions, keyframes)  # [batch_size, 5] in meter
                    kps_error = compute_kps_error_arbitrary(cur_motion, gt_skel_motions, keyframes_indices, traj_only=True)
                    keyframe_error = compute_kps_error_arbitrary(cur_motion, gt_skel_motions, keyframes_indices, traj_only=False)
                    skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]
                    # We can get the trajectory from here. Get only root xz from motion
                    cur_traj = cur_motion[:, 0, [0, 2], :]

                    # NOTE: To test if the motion is reasonable or not
                    if log_motion:
                        from data_loaders.humanml.utils.plot_script import plot_3d_motion
                        for j in tqdm([1, 3, 4, 5], desc="generating motion"):
                            motion_id = f'{i:04d}_{t:02d}_{j:02d}'
                            plot_3d_motion(os.path.join(self.save_dir, f"motion_cond_{motion_id}.mp4"), self.dataset.kinematic_chain,
                            cur_motion[j].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

                    if self.dataset.absolute_3d:
                        # NOTE: Changing the output from absolute space to the relative space here.
                        # The easiest way to do this is to go all the way to skeleton and convert back again.
                        # sample shape [32, 263, 1, 196]
                        sample = abs3d_to_rel(sample, self.dataset, motion_model)

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'dist_error': kps_error[bs_i].cpu().numpy(),
                                    'skate_ratio': skate_ratio[bs_i],
                                    'keyframe_error': keyframe_error[bs_i].cpu().numpy(),
                                    'num_keyframes': len(keyframes_indices[bs_i]) if keyframes_indices[bs_i] is not None else 0,
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        'traj': cur_traj[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        dist_error = data['dist_error']
        skate_ratio = data['skate_ratio']
        sent_len = data['cap_len']
        keyframe_error = data['keyframe_error']
        num_keyframes = data['num_keyframes']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            if self.dataset.absolute_3d:
                # Denorm with rel_transform because the inv_transform() will have the absolute mean and std
                # The motion is already converted to relative after inference
                denormed_motion = (normed_motion * self.dataset.std_rel) + self.dataset.mean_rel
            else:
                denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), dist_error, skate_ratio, keyframe_error, num_keyframes


    def get_keyframe_indices(self, keyframes_mask):
        keyframe_indices = []
        for sample_i in range(keyframes_mask.shape[0]):
            keyframe_indices.append([int(e) for e in keyframes_mask[sample_i].sum(dim=0).squeeze().nonzero().squeeze(-1)])
        return keyframe_indices


    def set_inference_editing_args(self, model_kwargs, input_motions):
        """ Set arguments for inference-time editing according to edit.py

        Args:
            model_kwargs (dict): arguments for the model
            input_motions (torch.tensor): ground-truth motion with absolute-root representation

        Returns:
            torch.tensor: keyframe_indices
            torch.tensor: joint_mask
        """
        model_kwargs['y']['inpainted_motion'] = input_motions
        model_kwargs['y']['imputate'] = self.args.imputate
        model_kwargs['y']['replacement_distribution'] = self.args.replacement_distribution
        model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
        model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
        model_kwargs['y']['diffusion_steps'] = self.args.diffusion_steps
        model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
        model_kwargs['y']['stop_imputation_at'] = self.args.stop_imputation_at
        model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at

        # if args.text_condition == '':
        #     args.guidance_param = 0.  # Force unconditioned generation

        model_kwargs['y']['inpainting_mask'], joint_mask = get_keyframes_mask(data=model_kwargs['y']['inpainted_motion'],
                                                                                lengths=model_kwargs['y']['lengths'],
                                                                                edit_mode=self.args.edit_mode,
                                                                                trans_length=self.args.transition_length,
                                                                                feature_mode=self.args.editable_features,
                                                                                get_joint_mask=True, n_keyframes=self.args.n_keyframes)

        return self.get_keyframe_indices(model_kwargs['y']['inpainting_mask']), joint_mask


    def set_conditional_synthesis_args(self, model_kwargs, input_motions):
        """ Set arguments for conditional sampling according to conditional_synthesis.py

        Args:
            model_kwargs (dict): arguments for the model
            input_motions (torch.tensor): ground-truth motion with absolute-root representation

        Returns:
            torch.tensor: keyframe_indices
            torch.tensor: joint_mask
        """
        model_kwargs['obs_x0'] = input_motions
        model_kwargs['obs_mask'], joint_mask = get_keyframes_mask(data=input_motions, lengths=model_kwargs['y']['lengths'], edit_mode=self.args.edit_mode,
                                                                  feature_mode=self.args.editable_features, trans_length=self.args.transition_length,
                                                                  get_joint_mask=True, n_keyframes=self.args.n_keyframes)
        model_kwargs['y']['diffusion_steps'] = self.args.diffusion_steps
        # Add inpainting mask according to args
        if self.args.zero_keyframe_loss: # if loss is 0 over keyframes durint training, then must impute keyframes during inference
            model_kwargs['y']['imputate'] = 1
            model_kwargs['y']['stop_imputation_at'] = 0
            model_kwargs['y']['replacement_distribution'] = 'conditional'
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask'] # used to do [nsamples, nframes] --> [nsamples, njoints, nfeats, nframes]
            model_kwargs['y']['reconstruction_guidance'] = False
        elif self.args.imputate: # if loss was present over keyframes during training, we may use inpaiting at inference time
            model_kwargs['y']['imputate'] = 1
            model_kwargs['y']['stop_imputation_at'] = self.args.stop_imputation_at
            model_kwargs['y']['replacement_distribution'] = 'conditional' # TODO: check if should also support marginal distribution
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
            if self.args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
                model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
                model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
                model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
                model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at
        elif self.args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
            model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
            model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
            model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
            model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at

        return self.get_keyframe_indices(model_kwargs['obs_mask']), joint_mask
