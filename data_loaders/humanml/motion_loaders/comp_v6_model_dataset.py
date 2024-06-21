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

from data_loaders.humanml.data.dataset import abs3d_to_rel, sample_to_motion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.metrics import calculate_skating_ratio
from sample.gmd.condition import (cond_fn_key_location, get_target_from_kframes, get_target_and_inpt_from_kframes_batch,
                              log_trajectory_from_xstart, get_inpainting_motion_from_traj, get_inpainting_motion_from_gt,
                              cond_fn_key_location, compute_kps_error, cond_fn_sdf,
                              CondKeyLocations, CondKeyLocationsWithSdf)


def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., save_dir=None, seed=None):
        assert seed is not None, "seed must be provided"
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.save_dir = save_dir
        assert save_dir is not None
        assert mm_num_samples < len(dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        clip_denoised = True
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

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

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    # setting seed here make sure that the same seed is used even continuing from unfinished runs
                    seed_number = seed * 100_000 + i * 100 + t
                    fixseed(seed_number)

                    batch_file = f'{i:04d}_{t:02d}.pt'
                    batch_path = os.path.join(self.save_dir, batch_file)

                    # reusing the batch if it exists
                    if os.path.exists(batch_path):
                        # [bs, njoints, nfeat, seqlen]
                        sample = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:
                        # [bs, njoints, nfeat, seqlen]
                        sample = sample_fn(
                            model,
                            motion.shape,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                        )
                        # save to file
                        torch.save(sample, batch_path)

                    # print('cut the motion length from {} to {}'.format(sample.shape[-1], self.max_motion_length))
                    sample = sample[:, :, :, :self.max_motion_length]
                    # Compute error for key xz locations
                    cur_motion = sample_to_motion(sample, self.dataset, model)
                    # We can get the trajectory from here. Get only root xz from motion
                    cur_traj = cur_motion[:, 0, [0, 2], :]

                    # NOTE: To test if the motion is reasonable or not
                    log_motion = False
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
                        sample = abs3d_to_rel(sample, self.dataset, model)

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
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
        sent_len = data['cap_len']
        if 'skate_ratio' in data.keys():
            skate_ratio = data['skate_ratio']
        else:
            skate_ratio = -1

        # print("get item")
        # print("abs ", self.dataset.absolute_3d)
        # print(self.dataset.mode)
        # if self.dataset.absolute_3d:
        #     # If we use the dataset with absolute 3D location, we need to convert the motion to relative first
        #     normed_motion = motion
        #     denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
        #     # Convert the denormed_motion from absolute 3D position to relative
        #     # denormed_motion_relative = self.dataset.t2m_dataset.abs3d_to_rel(denormed_motion)
        #     denormed_motion_relative = abs3d_to_rel(denormed_motion)

        #     if self.dataset.mode == 'eval':
        #         # Normalize again with the *T2M* mean and std
        #         renormed_motion = (denormed_motion_relative - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
        #         motion = renormed_motion
        #     else:
        #         # Normalize again with the *relative* mean and std.
        #         # Expect mode 'gt'
        #         # This assume that we will want to use this function to only get gt or for eval
        #         raise NotImplementedError
        #         renormed_motion_relative = (denormed_motion_relative - self.dataset.mean_rel) / self.dataset.std_rel
        #         motion = renormed_motion_relative

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

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), skate_ratio


# Data class for generated motion by *conditioning*
class CompMDMGeneratedDatasetCondition(Dataset):

    def __init__(self, model_dict, diffusion_dict, dataloader, mm_num_samples, mm_num_repeats,
                 max_motion_length, num_samples_limit, scale=1., save_dir=None, impute_until=0, skip_first_stage=False,
                 seed=None, use_ddim=False):

        assert seed is not None, "must provide seed"

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.save_dir = save_dir
        # This affect the trajectory model if we do two-stage, if not, it will affect the motion model
        # For trajectory model, the output traj will be imptued until 20 (set by impute_slack)
        self.impute_until = impute_until

        motion_model, traj_model = model_dict["motion"], model_dict["traj"]
        motion_diffusion, traj_diffusion = diffusion_dict["motion"], diffusion_dict["traj"]

        ### Basic settings
        motion_classifier_scale = 100.0
        print("motion classifier scale", motion_classifier_scale)
        log_motion = False
        guidance_mode = 'no'
        abs_3d = True
        use_random_proj = self.dataset.use_rand_proj
        print("guidance mode", guidance_mode)
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

        if traj_diffusion is not None:
            trajectory_classifier_scale = 100.0 # 100.0
            print("trajectory classifier scale", trajectory_classifier_scale)
            traj_diffusion.data_transform_fn = None
            traj_diffusion.data_inv_transform_fn = None
            if log_motion:
                traj_diffusion.log_trajectory_fn = partial(
                    log_trajectory_from_xstart,
                    kframes=[],
                    inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                    abs_3d=abs_3d,  # <--- assume the traj model is absolute
                    traject_only=True,
                    n_frames=max_motion_length)
            sample_fn_traj = (
                traj_diffusion.p_sample_loop if not use_ddim else traj_diffusion.ddim_sample_loop
            )
            traj_model.eval()
        else:
            # If we don't have a trajectory diffusion model, assume that we are using classifier-free 1-stage model
            pass

        assert save_dir is not None
        assert mm_num_samples < len(dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        # use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        clip_denoised = True
        self.max_motion_length = max_motion_length

        sample_fn_motion = (
            motion_diffusion.p_sample_loop if not use_ddim else motion_diffusion.ddim_sample_loop
        )

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
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                ### 1. Prepare motion for conditioning ###
                traj_model_kwargs = copy.deepcopy(model_kwargs)
                traj_model_kwargs['y']['traj_model'] = True
                model_kwargs['y']['traj_model'] = False

                # Convert to 3D motion space
                # NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_rand_proj' is True
                gt_poses = motion.permute(0, 2, 3, 1)
                gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263]
                # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=False)
                gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                                                    jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # gt_skel_motions shape [32, 22, 3, 196]
                # # Visualize to make sure it is correct
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # plot_3d_motion("./test_positions_1.mp4", self.dataset.kinematic_chain,
                #                gt_skel_motions[0].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

                # Next, sample points, then prepare target and inpainting mask for trajectory model
                ## Sample points
                n_keyframe = 5
                # reusing the target if it exists
                target_batch_file = f'target_{i:04d}.pt'
                target_batch_file = os.path.join(self.save_dir, target_batch_file)
                if os.path.exists(target_batch_file):
                    # [batch_size, n_keyframe]
                    sampled_keyframes = torch.load(target_batch_file, map_location=motion.device)
                    print(f'sample keyframes {target_batch_file} exists, loading from file')
                else:
                    sampled_keyframes = torch.rand(motion.shape[0], n_keyframe) * model_kwargs['y']['lengths'].unsqueeze(-1)
                    # Floor to int because ceil to 'lengths' will make the idx out-of-bound.
                    # The keyframe can be a duplicate.
                    sampled_keyframes = torch.floor(sampled_keyframes).int().sort()[0]  # shape [batch_size, n_keyframe]
                    torch.save(sampled_keyframes, target_batch_file)
                # import pdb; pdb.set_trace()
                ## Prepare target and mask for grad cal
                # Prepare trajecotry inpainting
                (target, target_mask,
                 inpaint_traj, inpaint_traj_mask,
                 inpaint_traj_points, inpaint_traj_mask_points,
                 inpaint_motion, inpaint_mask,
                 inpaint_motion_points, inpaint_mask_points) = get_target_and_inpt_from_kframes_batch(gt_skel_motions, sampled_keyframes, self.dataset)

                target = target.to(model_device)
                target_mask = target_mask.to(model_device)
                model_kwargs['y']['target'] = target
                model_kwargs['y']['target_mask'] = target_mask
                # target [32, 196, 22, 3]  # in 3d skeleton
                # inpaint [32, 4, 1, 196]  # in model input space
                ### End 1. preparing condition ###

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
                    if os.path.exists(batch_path):
                        # [bs, njoints, nfeat, seqlen]
                        sample_motion = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:
                        print(f'working on {batch_file}')
                        # for smoother motions
                        impute_slack = 20
                        # NOTE: For debugging
                        traj_model_kwargs['y']['log_name'] = self.save_dir
                        traj_model_kwargs['y']['log_id'] = i
                        model_kwargs['y']['log_name'] = self.save_dir
                        model_kwargs['y']['log_id'] = i
                        # motion model always impute until 20
                        model_kwargs['y']['cond_until'] = impute_slack
                        model_kwargs['y']['impute_until'] = impute_slack

                        if skip_first_stage:
                            # No first stage. Skip straight to second stage
                            ### Add motion to inpaint
                            # import pdb; pdb.set_trace()
                            # del model_kwargs['y']['inpainted_motion']
                            # del model_kwargs['y']['inpainting_mask']
                            model_kwargs['y']['inpainted_motion'] = inpaint_motion.to(model_device) # init_motion.to(model_device)
                            model_kwargs['y']['inpainting_mask'] = inpaint_mask.to(model_device)

                            model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_motion_points.to(model_device)
                            model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_mask_points.to(model_device)
                            # import pdb; pdb.set_trace()

                            # For classifier-free
                            CLASSIFIER_FREE = True
                            if CLASSIFIER_FREE:
                                impute_until = 1
                                impute_slack = 20
                                # del model_kwargs['y']['inpainted_motion']
                                # del model_kwargs['y']['inpainting_mask']
                                model_kwargs['y']['inpainted_motion'] = inpaint_motion_points.to(model_device) # init_motion.to(model_device)
                                model_kwargs['y']['inpainting_mask'] = inpaint_mask_points.to(model_device)

                            # Set when to stop imputing
                            model_kwargs['y']['cond_until'] = impute_slack
                            model_kwargs['y']['impute_until'] = impute_until
                            model_kwargs['y']['impute_until_second_stage'] = impute_slack

                        else:
                            ### Add motion to inpaint
                            traj_model_kwargs['y']['inpainted_motion'] = inpaint_traj.to(model_device) # init_motion.to(model_device)
                            traj_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask.to(model_device)

                            # Set when to stop imputing
                            traj_model_kwargs['y']['cond_until'] = impute_slack
                            traj_model_kwargs['y']['impute_until'] = impute_until
                            # NOTE: We have the option of switching the target motion from line to just key locations
                            # We call this a 'second stage', which will start after t reach 'impute_until'
                            traj_model_kwargs['y']['impute_until_second_stage'] = impute_slack
                            traj_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points.to(model_device)
                            traj_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points.to(model_device)


                            ##########################################################
                            # print("************* Test: not using dense gradient ****************")
                            # NO_GRAD = True
                            # traj_model_kwargs['y']['cond_until'] = 1000

                            # traj_model_kwargs['y']['impute_until'] = 1000
                            # traj_model_kwargs['y']['impute_until_second_stage'] = 0

                            ##########################################################

                            ### Generate trajectory
                            # [bs, njoints, nfeat, seqlen]
                            # NOTE: add cond_fn
                            sample_traj = sample_fn_traj(
                                traj_model,
                                inpaint_traj.shape,
                                clip_denoised=clip_denoised,
                                model_kwargs=traj_model_kwargs,  # <-- traj_kwards
                                skip_timesteps=0,  # NOTE: for debugging, start from 900
                                init_image=None,
                                progress=True,
                                dump_steps=None,
                                noise=None,
                                const_noise=False,
                                cond_fn=partial(
                                    cond_fn_key_location, # cond_fn_sdf, #,
                                    transform=self.dataset.t2m_dataset.transform_th,
                                    inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                                    target=target,
                                    target_mask=target_mask,
                                    kframes=[],
                                    abs_3d=abs_3d, # <<-- hard code,
                                    classifiler_scale=trajectory_classifier_scale,
                                    use_mse_loss=False),  # <<-- hard code
                            )

                            ### Prepare conditions for motion from generated trajectory ###
                            # Get inpainting information for motion model
                            traj_motion, traj_mask = get_inpainting_motion_from_traj(
                                sample_traj, inv_transform_fn=self.dataset.t2m_dataset.inv_transform_th)
                            # Get target for loss grad
                            # Target has dimention [bs, max_motion_length, 22, 3]
                            target = torch.zeros([motion.shape[0], max_motion_length, 22, 3], device=traj_motion.device)
                            target_mask = torch.zeros_like(target, dtype=torch.bool)
                            # This assume that the traj_motion is in the 3D space without normalization
                            # traj_motion: [3, 263, 1, 196]
                            target[:, :, 0, [0, 2]] = traj_motion.permute(0, 3, 2, 1)[:, :, 0,[1, 2]]
                            target_mask[:, :, 0, [0, 2]] = True
                            # Set imputing trajectory
                            model_kwargs['y']['inpainted_motion'] = traj_motion
                            model_kwargs['y']['inpainting_mask'] = traj_mask
                            ### End - Prepare conditions ###

                        # import pdb; pdb.set_trace()

                        ### Generate motion
                        # NOTE: add cond_fn
                        # TODO: move the followings to a separate function
                        if guidance_mode == "kps" or guidance_mode == "trajectory":
                            cond_fn = CondKeyLocations(target=target,
                                                        target_mask=target_mask,
                                                        transform=self.dataset.t2m_dataset.transform_th,
                                                        inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                                                        abs_3d=abs_3d,
                                                        classifiler_scale=motion_classifier_scale,
                                                        use_mse_loss=False,
                                                        use_rand_projection=self.dataset.use_random_proj
                                                        )
                        # elif guidance_mode == "sdf":
                        #     cond_fn = CondKeyLocationsWithSdf(target=target,
                        #                                 target_mask=target_mask,
                        #                                 transform=data.dataset.t2m_dataset.transform_th,
                        #                                 inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                        #                                 abs_3d=abs_3d,
                        #                                 classifiler_scale=motion_classifier_scale,
                        #                                 use_mse_loss=False,
                        #                                 use_rand_projection=self.dataset.use_random_proj,
                        #                                 obs_list=obs_list
                        #                                 )
                        elif guidance_mode == "no" or guidance_mode == "mdm_legacy":
                            cond_fn = None

                        # if NO_GRAD:
                        #     cond_fn = None

                        sample_motion = sample_fn_motion(
                            motion_model,
                            (motion.shape[0], motion_model.njoints, motion_model.nfeats, motion.shape[3]),  # motion.shape
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            cond_fn=cond_fn
                                # partial(
                                # cond_fn_key_location,
                                # transform=self.dataset.t2m_dataset.transform_th,
                                # inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                                # target=target,
                                # target_mask=target_mask,
                                # kframes=[],
                                # abs_3d=True, # <<-- hard code,
                                # classifiler_scale=motion_classifier_scale,
                                # use_mse_loss=False),  # <<-- hard code
                        )
                        # save to file
                        torch.save(sample_motion, batch_path)


                    # print('cut the motion length from {} to {}'.format(sample_motion.shape[-1], self.max_motion_length))
                    sample = sample_motion[:, :, :, :self.max_motion_length]

                    # Compute error for key xz locations
                    cur_motion = sample_to_motion(sample, self.dataset, motion_model)
                    kps_error = compute_kps_error(cur_motion, gt_skel_motions, sampled_keyframes)  # [batch_size, 5] in meter
                    skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]
                    # import pdb; pdb.set_trace()
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
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        'traj': cur_traj[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]
                        # import pdb; pdb.set_trace()

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

        if self.dataset.mode == 'eval':
            normed_motion = motion
            if self.dataset.absolute_3d:
                # Denorm with rel_transform because the inv_transform() will have the absolute mean and std
                # The motion is already converted to relative after inference
                # import pdb; pdb.set_trace()
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

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), dist_error, skate_ratio


# Data class for generated motion by *inpainting full trajectory*
class CompMDMGeneratedDatasetInpainting(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., save_dir=None, seed=None):
        assert seed is not None, "seed must be provided"
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.save_dir = save_dir
        assert save_dir is not None
        assert mm_num_samples < len(dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Settings
        motion_classifier_scale = 100.0
        print("motion classifier scale", motion_classifier_scale)
        log_motion = False # False

        model_device = next(model.parameters()).device
        diffusion.data_get_mean_fn = self.dataset.t2m_dataset.get_std_mean
        diffusion.data_transform_fn = self.dataset.t2m_dataset.transform_th
        diffusion.data_inv_transform_fn = self.dataset.t2m_dataset.inv_transform_th
        if log_motion:
            diffusion.log_trajectory_fn = partial(
                log_trajectory_from_xstart,
                kframes=[],
                inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                abs_3d=True,  # <--- assume the motion model is absolute
                use_rand_proj=self.dataset.use_rand_proj,
                traject_only=False,
                n_frames=max_motion_length)

        use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        clip_denoised = True
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

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
        model.eval()

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                model_kwargs['y']['log_name'] = self.save_dir
                ### 1. Prepare motion for conditioning ###
                model_kwargs['y']['traj_model'] = False
                model_kwargs['y']['log_id'] = i
                # Convert to 3D motion space
                # NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_rand_proj' is True
                gt_poses = motion.permute(0, 2, 3, 1)
                gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263]
                # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=False)
                gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                gt_skel_motions = model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                                                    jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # gt_skel_motions shape [32, 22, 3, 196]
                # # Visualize to make sure it is correct
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # plot_3d_motion("./test_positions_1.mp4", self.dataset.kinematic_chain,
                #                gt_skel_motions[0].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

                ## Prepare target and mask for grad cal
                inpaint_motion, inpaint_mask, target, target_mask = get_inpainting_motion_from_gt(
                    gt_skel_motions, dataloader.batch_size, model_device, model_kwargs['y']['lengths'],
                    inv_transform_fn=self.dataset.t2m_dataset.inv_transform_th)
                model_kwargs['y']['target'] = target
                model_kwargs['y']['target_mask'] = target_mask
                # target [32, 196, 22, 3]  # in 3d skeleton
                # inpaint [32, 263, 1, 196]  # in model input space
                ### End 1. preparing condition ###

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    # setting seed here make sure that the same seed is used even continuing from unfinished runs
                    seed_number = seed * 100_000 + i * 100 + t
                    fixseed(seed_number)

                    batch_file = f'{i:04d}_{t:02d}.pt'
                    batch_path = os.path.join(self.save_dir, batch_file)

                    # reusing the batch if it exists
                    if os.path.exists(batch_path):
                        # [bs, njoints, nfeat, seqlen]
                        sample = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:
                        # Set inpainting information
                        model_kwargs['y']['inpainted_motion'] = inpaint_motion.to(model_device)
                        model_kwargs['y']['inpainting_mask'] = inpaint_mask.to(model_device)
                        # Set when to stop imputing
                        model_kwargs['y']['impute_until'] = 0
                        model_kwargs['y']['cond_until'] = 0

                        # [bs, njoints, nfeat, seqlen]
                        do_optimize = False
                        if do_optimize:
                            cond_fn = partial(
                                cond_fn_key_location,
                                transform=self.dataset.t2m_dataset.transform_th,
                                inv_transform=self.dataset.t2m_dataset.inv_transform_th,
                                target=target,
                                target_mask=target_mask,
                                kframes=[],
                                abs_3d=True, # <<-- hard code,
                                classifiler_scale=motion_classifier_scale,
                                use_mse_loss=False)  # <<-- hard code
                        else:
                            cond_fn = None
                        sample = sample_fn(
                            model,
                            (motion.shape[0], model.njoints, model.nfeats, motion.shape[3]),  # motion.shape
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            cond_fn=cond_fn,
                        )
                        # save to file
                        torch.save(sample, batch_path)

                    # print('cut the motion length from {} to {}'.format(sample.shape[-1], self.max_motion_length))
                    sample = sample[:, :, :, :self.max_motion_length]
                    cur_motion = sample_to_motion(sample, self.dataset, model)
                    skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]

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
                        sample = abs3d_to_rel(sample, self.dataset, model)

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'skate_ratio': skate_ratio[bs_i],
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
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
        sent_len = data['cap_len']
        skate_ratio = data['skate_ratio']

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

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), skate_ratio