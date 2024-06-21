import numpy as np
import os
import numpy as np
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from argparse import ArgumentParser


def plot_samples(motions, gt_motions, lengths, texts, out_path, all_observed_masks=None):
    fps = 10 # TODO: only for debugging purposes, reduce fps. Remove line later.
    skeleton = paramUtil.t2m_kinematic_chain
    for sample_i in range(motions.shape[0]):
        caption = 'GT Motion - {}'.format(texts[sample_i])
        length = int(lengths[sample_i])
        motion = gt_motions[sample_i].numpy().transpose(2, 0, 1)[:length]
        save_file = 'gt_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                       dataset='humanml', fps=fps, vis_mode='gt')

        caption = 'Sample - {}'.format(texts[sample_i])
        motion = motions[sample_i].numpy().transpose(2, 0, 1)[:length]
        save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, 0)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({sample_i}) "{caption}" -> {save_file}]')

        gt_frames = np.where(all_observed_masks[sample_i, 0, 0, :])[0] if all_observed_masks is not None else []
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                       dataset='humanml', fps=fps, vis_mode='in_between', gt_frames=gt_frames)

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={1+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def plot_sample(motions, gt_motions, lengths, out_path):
    fps = 10 # TODO: only for debugging purposes, reduce fps. Remove line later.
    skeleton = paramUtil.t2m_kinematic_chain
    for idx in range(motions.shape[0]):
        save_path = os.path.join(out_path, f'sample_{idx}.mp4')
        length = int(lengths[idx])
        motion = motions[idx].numpy().transpose(2, 0, 1)[:length]
        gt_motion = gt_motions[idx].numpy().transpose(2, 0, 1)[:length]
        plot_3d_motion(save_path, skeleton, motion, dataset='humanml', title='Sampled Motion', fps=fps)
        plot_3d_motion(save_path, skeleton, gt_motion, dataset='humanml', title='GT Motion', fps=fps)


def plot_conditional_samples(motion, lengths, texts, observed_motion, observed_mask, num_samples, num_repetitions, out_path, edit_mode='benchmark_sparse', stop_imputation_at=0):
    '''
    Used to plot samples during conditionally keyframed training.
    Arguments:
        motion {torch.Tensor} -- sampled batch of motions (nreps, nsamples, 22, 3, nframes)
        lengths {torch.Tensor} -- motion lengths (nreps, nsamples)
        texts {torch.Tensor} -- texts of motions (nreps * nsamples)
        observed_motion {torch.Tensor} -- ground-truth motions (nsamples, 22, 3, nframes)
        observed_mask {torch.Tensor} -- keyframes mask (nsamples, 22, 3, nframes)
        cutoff {int} -- if any replacement, set cutoff to 0 otherwise a value larger than 0
    Returns:
        matplotlib.pyplot.subplots -- figure
    '''

    dataset = 'humanml'
    batch_size = num_samples

    fps = 10 # TODO: only for debugging purposes, reduce fps. Remove line later.
    skeleton = paramUtil.t2m_kinematic_chain
    for sample_i in range(num_samples):
        caption = 'Input Motion'
        length = lengths[0, sample_i]
        gt_motion = observed_motion[sample_i].transpose(2, 0, 1)[:length]
        save_file = 'input_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, gt_motion, title=caption,
                        dataset=dataset, fps=fps, vis_mode='gt',
                        gt_frames=np.where(observed_mask[sample_i, 0, 0, :])[0])
        for rep_i in range(num_repetitions):
            caption = texts[rep_i * batch_size + sample_i]
            if caption == '':
                caption = 'Edit [{}] unconditioned'.format(edit_mode)
            else:
                caption = 'Edit [{}]: {}'.format(edit_mode, caption)
            length = lengths[rep_i, sample_i]
            gen_motion = motion[rep_i, sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            vis_mode = edit_mode if edit_mode in ['upper_body', 'pelvis', 'right_wrist', 'pelvis_feet', 'pelvis_vr'] else 'benchmark_sparse'
            gt_frames = [] if edit_mode in ['upper_body', 'pelvis', 'right_wrist', 'pelvis_feet', 'pelvis_vr'] else np.where(observed_mask[sample_i, 0, 0, :])[0]
            plot_3d_motion(animation_save_path, skeleton, gen_motion, title=caption,
                        dataset=dataset, fps=fps, vis_mode=vis_mode,
                        gt_frames=gt_frames)

            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saved_results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    results = np.load(os.path.join(args.saved_results_dir, 'results.npy'), allow_pickle=True).item()

    motion = results['motion']
    texts = results['text']
    lengths = results['lengths']
    num_samples = results['num_samples']
    num_repetitions = results['num_repetitions']
    observed_motion = results['observed_motion']
    observed_mask = results['observed_mask']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot_conditional_samples(motion=results['motion'],
                             lengths=results['lengths'],
                             texts=results['text'],
                             observed_motion=results['observed_motion'],
                             observed_mask=results['observed_mask'],
                             num_samples=results['num_samples'],
                             num_repetitions=results['num_repetitions'],
                             out_path=args.output_dir,
                             edit_mode='benchmark_sparse', #FIXME: only works for selected edit modes.
                             cutoff=0) #FIXME: set to 0 for now to always replace with ground-truth keyframes --> mainly for visualization purposes.