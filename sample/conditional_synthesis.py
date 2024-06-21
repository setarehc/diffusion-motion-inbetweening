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
from data_loaders.get_data import get_dataset_loader, DatasetConfig
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from pathlib import Path
from utils.editing_util import get_keyframes_mask, load_fixed_dataset
from data_loaders.humanml.utils.plotting import plot_conditional_samples
import json


def compute_editing_mask(data, lengths, args):
    # assert args.edit_feature_mode == args.feature_mode
    # always use features seen during training
    if args.keyframe_selection_scheme in ['random_frames']:
        # Model is trained conditionally using a general keyframing scheme
        # In this case, we can choose the keyframes according to input arguments matching MDM inbetweening or NeMF benchmarks
        assert args.edit_mode in ['in_between', 'benchmark_sparse', 'benchmark_clip']
        return get_keyframes_mask(data=data, lengths=lengths, edit_mode=args.edit_mode,
                                  feature_mode=args.feature_mode, trans_length=args.transition_length,
                                  get_joint_mask=True)
    elif args.inbetween_mode == 'nemf':
        # Model is trained with randomly sampled NeMF keyframing schemes
        # In this case, we can choose the keyframes according to input arguments matching NeMF bencmarks
        assert args.edit_mode in ['benchmark_sparse', 'benchmark_clip']
        return get_keyframes_mask(data=data, lengths=lengths, edit_mode=args.edit_mode, feature_mode=args.feature_mode,
                                  trans_length=args.transition_length, get_joint_mask=True)

    elif args.inbetween_mode == 'random_joints':
        # Model is most flexible here
        return get_keyframes_mask(data=data, lengths=lengths, edit_mode=args.edit_mode, feature_mode=args.feature_mode,
                                  trans_length=args.transition_length, get_joint_mask=True)
    else:
        # Model is trained with one of these keyframing schemes: ['inbetween', 'benchmark_sparse', 'benchmark_clip', 'upper_body', 'pelvis']
        # In this case, we must choose the keyframes similar to the training scheme
        return get_keyframes_mask(data=data, lengths=lengths, edit_mode=args.inbetween_mode,
                                  feature_mode=args.feature_mode, trans_length=args.transition_length,
                                  get_joint_mask=True)


def main():
    args = cond_synt_args()
    fixseed(args.seed)

    assert args.dataset == 'humanml' and args.abs_3d # Only humanml dataset and the absolute root representation is supported for conditional synthesis
    assert args.keyframe_conditioned

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else (200 if args.dataset == 'trajectories' else 60)
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        checkpoint_name = os.path.split(os.path.dirname(args.model_path))[-1]
        model_results_path = os.path.join('save/results', checkpoint_name)
        method = ''
        if args.imputate:
            method += '_' + 'imputation'
        if args.reconstruction_guidance:
            method += '_' + 'recg'

        if args.editable_features != 'pos_rot_vel':
            edit_mode = args.edit_mode + '_' + args.editable_features
        else:
            edit_mode = args.edit_mode
        out_path = os.path.join(model_results_path,
                                'condsamples{}_{}_{}_T={}_CI={}_CRG={}_KGP={}_seed{}'.format(niter, method,
                                                                                      edit_mode, args.transition_length,
                                                                                      args.stop_imputation_at, args.stop_recguidance_at,
                                                                                      args.keyframe_guidance_param, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    use_test_set_prompts = False
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
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
    elif args.no_text:
        texts = [''] * args.num_samples
        args.guidance_param = 0.  # Force unconditioned generation # TODO: This is part of inbetween.py --> Will I need it here?
    else:
        # use text from the test set
        use_test_set_prompts = True

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples # Sampling a single batch from the testset, with exactly args.num_samples
    split = 'fixed_subset' if args.use_fixed_subset else 'test'
    data = load_dataset(args, max_frames, split=split)

    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)
    if args.guidance_param != 1 and args.keyframe_guidance_param != 1:
        raise NotImplementedError('Classifier-free sampling for keyframes not implemented.')
    elif args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################

    iterator = iter(data)

    input_motions, model_kwargs = next(iterator)

    if args.use_fixed_dataset: # TODO: this is for debugging - need a neater way to do this for the final version - num_samples should be 10
        assert args.dataset == 'humanml' and args.abs_3d
        input_motions, model_kwargs = load_fixed_dataset(args.num_samples)

    input_motions = input_motions.to(dist_util.dev()) # [nsamples, njoints=263/1, nfeats=1/3, nframes=196/200]
    input_masks = model_kwargs["y"]["mask"]  # [nsamples, 1, 1, nframes]
    input_lengths = model_kwargs["y"]["lengths"]  # [nsamples]

    model_kwargs['obs_x0'] = input_motions
    model_kwargs['obs_mask'], obs_joint_mask = get_keyframes_mask(data=input_motions, lengths=input_lengths, edit_mode=args.edit_mode,
                                                                  feature_mode=args.editable_features, trans_length=args.transition_length,
                                                                  get_joint_mask=True, n_keyframes=args.n_keyframes) # [nsamples, njoints, nfeats, nframes]

    assert max_frames == input_motions.shape[-1]

    # Arguments
    model_kwargs['y']['text'] = texts if not use_test_set_prompts else model_kwargs['y']['text']
    model_kwargs['y']['diffusion_steps'] = args.diffusion_steps

    # Add inpainting mask according to args
    if args.zero_keyframe_loss: # if loss is 0 over keyframes durint training, then must impute keyframes during inference
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = 0
        model_kwargs['y']['replacement_distribution'] = 'conditional'
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask'] # used to do [nsamples, nframes] --> [nsamples, njoints, nfeats, nframes]
        model_kwargs['y']['reconstruction_guidance'] = False
    elif args.imputate: # if loss was present over keyframes during training, we may use imputation at inference time
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = args.stop_imputation_at
        model_kwargs['y']['replacement_distribution'] = 'conditional' # TODO: check if should also support marginal distribution
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
        if args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
            model_kwargs['y']['reconstruction_guidance'] = args.reconstruction_guidance
            model_kwargs['y']['reconstruction_weight'] = args.reconstruction_weight
            model_kwargs['y']['gradient_schedule'] = args.gradient_schedule
            model_kwargs['y']['stop_recguidance_at'] = args.stop_recguidance_at
    elif args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
        model_kwargs['y']['reconstruction_guidance'] = args.reconstruction_guidance
        model_kwargs['y']['reconstruction_weight'] = args.reconstruction_weight
        model_kwargs['y']['gradient_schedule'] = args.gradient_schedule
        model_kwargs['y']['stop_recguidance_at'] = args.stop_recguidance_at

    all_motions = []
    all_lengths = []
    all_text = []
    all_observed_motions = []
    all_observed_masks = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            # text classifier-free guidance
            model_kwargs['y']['text_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        if args.keyframe_guidance_param != 1:
            # keyframe classifier-free guidance
            model_kwargs['y']['keyframe_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.keyframe_guidance_param

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
            ) # [nsamples, njoints, nfeats, nframes]

        # Unnormalize samples and recover XYZ *positions*
        if model.data_rep == 'hml_vec':
            n_joints = 22 if (sample.shape[1] in [263, 264]) else 21
            sample = sample.cpu().permute(0, 2, 3, 1)
            sample = data.dataset.t2m_dataset.inv_transform(sample).float()
            sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # batch_size, n_joints=22, 3, n_frames

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        print(f"created {len(all_motions) * args.batch_size} samples")
        # Sampling is done!

    # Unnormalize observed motions and recover XYZ *positions*
    if model.data_rep == 'hml_vec':
        input_motions = input_motions.cpu().permute(0, 2, 3, 1)
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
        input_motions = recover_from_ric(data=input_motions, joints_num=n_joints, abs_3d=args.abs_3d)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1)
        input_motions = input_motions.cpu().numpy()
        inpainting_mask = obs_joint_mask.cpu().numpy()

    all_motions = np.stack(all_motions) # [num_rep, num_samples, 22, 3, n_frames]
    all_text = np.stack(all_text) # [num_rep, num_samples]
    all_lengths = np.stack(all_lengths) # [num_rep, num_samples]
    all_observed_motions = input_motions # [num_samples, 22, 3, n_frames]
    all_observed_masks = inpainting_mask

    os.makedirs(out_path, exist_ok=True)

    # Write run arguments to json file an save in out_path
    with open(os.path.join(out_path, 'edit_args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    npy_path = os.path.join(out_path, f'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
             'observed_motion': all_observed_motions, 'observed_mask': all_observed_masks})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text)) # TODO: Fix this for datasets other thah trajectories
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    if args.dataset == 'humanml':
        plot_conditional_samples(motion=all_motions,
                                 lengths=all_lengths,
                                 texts=all_text,
                                 observed_motion=all_observed_motions,
                                 observed_mask=all_observed_masks,
                                 num_samples=args.num_samples,
                                 num_repetitions=args.num_repetitions,
                                 out_path=out_path,
                                 edit_mode=args.edit_mode, #FIXME: only works for selected edit modes
                                 stop_imputation_at=0)


def load_dataset(args, max_frames, split='test'):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=split,
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