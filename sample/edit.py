# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import cond_synt_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil

import json
from utils.editing_util import get_keyframes_mask, load_fixed_dataset


def main():
    args = cond_synt_args()
    fixseed(args.seed)

    assert args.imputate or args.reconstruction_guidance, 'Edit mode requires either imputation or reconstruction guidance'

    out_path = args.output_dir
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        checkpoint_name = os.path.split(os.path.dirname(args.model_path))[-1]
        model_results_path = os.path.join('save/results', checkpoint_name)
        method = 'imputation_recg' if args.imputate and args.reconstruction_guidance else 'imputation' if args.imputate else 'recg'

        out_path = os.path.join(model_results_path,
                                'edit_{}_{}_{}_T={}_CI={}_CRG={}_seed{}'.format(niter, method,
                                                                                args.edit_mode, args.transition_length,
                                                                                args.stop_imputation_at, args.stop_recguidance_at,
                                                                                args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    # Create output directory
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Write run arguments to json file an save in out_path
    with open(os.path.join(out_path, 'edit_args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = load_dataset(args, max_frames)

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

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)

    if args.use_fixed_dataset: # TODO: this is for debugging - need a neater way to do this for the final version - num_samples should be 10
        assert args.dataset == 'humanml' and args.abs_3d
        input_motions, model_kwargs = load_fixed_dataset(args.num_samples)

    input_motions = input_motions.to(dist_util.dev())
    input_masks = model_kwargs['y']['mask']
    input_lengths = model_kwargs['y']['lengths']
    texts = [args.text_condition] * args.num_samples
    model_kwargs['y']['text'] = texts

    # Editing arguments:
    model_kwargs['y']['imputate'] = args.imputate
    model_kwargs['y']['replacement_distribution'] = args.replacement_distribution
    model_kwargs['y']['reconstruction_guidance'] = args.reconstruction_guidance
    model_kwargs['y']['reconstruction_weight'] = args.reconstruction_weight
    model_kwargs['y']['diffusion_steps'] = args.diffusion_steps
    model_kwargs['y']['gradient_schedule'] = args.gradient_schedule
    model_kwargs['y']['stop_imputation_at'] = args.stop_imputation_at
    model_kwargs['y']['stop_recguidance_at'] = args.stop_recguidance_at

    if args.text_condition == '':
        args.guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'], joint_mask = get_keyframes_mask(data=input_motions,
                                                                          lengths=input_lengths,
                                                                          edit_mode=args.edit_mode,
                                                                          trans_length=args.transition_length,
                                                                          feature_mode=args.editable_features,
                                                                          get_joint_mask=True,
                                                                          n_keyframes=args.n_keyframes)

    all_motions = []
    all_lengths = []
    all_text = []
    all_observed_motions = []
    all_observed_masks = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['text_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(data=sample, joints_num=n_joints, abs_3d=args.abs_3d)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")
    # Sampling is done!

    # Unnormalize observed motions and recover XYZ *positions*
    if model.data_rep == 'hml_vec':
        input_motions = input_motions.cpu().permute(0, 2, 3, 1)
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
        input_motions = recover_from_ric(data=input_motions, joints_num=n_joints, abs_3d=args.abs_3d)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1)
        input_motions = input_motions.cpu().numpy()
        inpainting_mask = joint_mask.cpu().numpy()

    all_motions = np.stack(all_motions) # [num_rep, num_samples, 22, 3, n_frames]
    all_text = np.stack(all_text) # [num_rep, num_samples]
    all_lengths = np.stack(all_lengths) # [num_rep, num_samples]
    all_observed_motions = input_motions # [num_samples, 22, 3, n_frames]
    all_observed_masks = inpainting_mask

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {'motion': all_motions,
                       'text': all_text,
                       'lengths': all_lengths,
                       'num_samples': args.num_samples,
                       'num_repetitions': args.num_repetitions,
                       'observed_motion': all_observed_motions,
                       'observed_mask': all_observed_masks})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    fps = 10 # TODO: only for debugging purposes. Remove this line later
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    for sample_i in range(args.num_samples):
        caption = 'Input Motion'
        length = model_kwargs['y']['lengths'][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = 'input_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                       dataset=args.dataset, fps=fps, vis_mode='gt',
                       gt_frames=np.where(all_observed_masks[sample_i, 0, 0, :])[0])
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            if caption == '':
                caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
            else:
                caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
            length = all_lengths[rep_i, sample_i]
            motion = all_motions[rep_i, sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            vis_mode = args.edit_mode if args.edit_mode in ['lower_body', 'pelvis'] else 'benchmark_sparse'
            gt_frames = [] if args.edit_mode in ['lower_body'] else np.where(all_observed_masks[sample_i, 0, 0, :])[0]
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=vis_mode,
                           gt_frames=gt_frames)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, max_frames):
        conf = DatasetConfig(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=max_frames,
            split='test',
            hml_mode='train',  # in train mode, you get both text and motion.
            use_abs3d=args.abs_3d,
            traject_only=args.traj_only,
            use_random_projection=args.use_random_proj,
            random_projection_scale=args.random_proj_scale,
            augment_type='none',
            std_scale_shift=args.std_scale_shift,
            drop_redundant=args.drop_redundant,
        )
        data = get_dataset_loader(conf)
        return data


if __name__ == "__main__":
    main()
