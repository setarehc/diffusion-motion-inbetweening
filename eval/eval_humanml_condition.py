from utils.parser_util import eval_args # , evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader, get_mdm_loader_cond  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_saved_model

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
import copy
# import flag
# import flag_traj

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        skate_ratio_sum = 0.0
        traj_err = []
        traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if motion_loader_name == "vald":
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, dist_error, skate_ratio = batch
                else:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch

                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens)
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

                if motion_loader_name == "vald":
                    # Compute dist error metrics
                    err_np = calculate_trajectory_error(dist_error)
                    traj_err.append(err_np)
                    skate_ratio_sum += skate_ratio.sum()


            all_motion_embeddings = np.concatenate(all_motion_embeddings,
                                                   axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings
        
        
        if motion_loader_name == "vald":
            ### For trajecotry evaluation ###
            traj_err = np.stack(traj_err).mean(0)
            trajectory_score_dict[motion_loader_name] = traj_err
            line = f'---> [{motion_loader_name}] Trajectory Error: '
            for (k, v) in zip(traj_err_key, traj_err):
                line += '(%s): %.4f ' % (k, np.mean(v))
            print(line)
            print(line, file=file, flush=True)

            # For skating evaluation
            skating_score = skate_ratio_sum / all_size
            skating_ratio_dict[motion_loader_name] = skating_score
            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
        ### ###

        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}'
        )
        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}',
            file=file,
            flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict, trajectory_score_dict, skating_ratio_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid

    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}',
              file=file,
              flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file,
                           mm_num_times):
    eval_dict = OrderedDict({})
    traj_diversity_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        mm_trajs = []
        motion_lengths = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens, trajs = batch
                # trajs [1, 2, 196, 2]
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
                mm_trajs.append(trajs)
                motion_lengths.append(m_lens[0, 0])

        # [32]
        motion_lengths = torch.stack(motion_lengths)
        # Calculate trajectory diversity
        # [32, 2, 196, 2 (xz)]
        mm_trajs = torch.cat(mm_trajs, dim=0).cpu().numpy()
        traj_diversity = calculate_trajectory_diversity(mm_trajs, motion_lengths)

        print(f'---> [{model_name}] Trajectory Diversity: {traj_diversity:.4f}')
        print(f'---> [{model_name}] Trajectory Diversity: {traj_diversity:.4f}',
              file=file,
              flush=True)
        traj_diversity_dict[model_name] = traj_diversity
        # End trajectory diversity

        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings,
                                             dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings,
                                                    mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}',
              file=file,
              flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict, traj_diversity_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper,
               gt_loader,
               eval_motion_loaders,
               log_file,
               replication_times,
               diversity_times,
               mm_num_times,
               run_mm=False):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'Trajectory Error': OrderedDict({}),
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
            'Trajectory Diversity': OrderedDict({}),
            'Skating Ratio': OrderedDict({})
        })
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items(
            ):
                # NOTE: set the seed for each motion loader based on the replication number
                motion_loader, mm_motion_loader = motion_loader_getter(
                    seed=replication)
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(
                f'==================== Replication {replication} ===================='
            )
            print(
                f'==================== Replication {replication} ====================',
                file=f,
                flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            (mat_score_dict, R_precision_dict, acti_dict, 
             trajectory_score_dict, skating_ratio_dict) = evaluate_matching_score(
                eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict, traj_diversity_dict = evaluate_multimodality(eval_wrapper,
                                                       mm_motion_loaders, f,
                                                       mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in trajectory_score_dict.items():
                if key not in all_metrics['Trajectory Error']:
                    all_metrics['Trajectory Error'][key] = [item]
                else:
                    all_metrics['Trajectory Error'][key] += [item]
            
            for key, item in skating_ratio_dict.items():
                if key not in all_metrics['Skating Ratio']:
                    all_metrics['Skating Ratio'][key] = [item]
                else:
                    all_metrics['Skating Ratio'][key] += [item]

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

                for key, item in traj_diversity_dict.items():
                    if key not in all_metrics['Trajectory Diversity']:
                        all_metrics['Trajectory Diversity'][key] = [item]
                    else:
                        all_metrics['Trajectory Diversity'][key] += [item]

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name,
                  file=f,
                  flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(
                        mean, np.float32):
                    print(
                        f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}'
                    )
                    print(
                        f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}',
                        file=f,
                        flush=True)
                elif metric_name == 'Trajectory Error':
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)): # zip(traj_err_key, mean):
                        line += '(%s): Mean: %.4f CInt: %.4f; ' % (traj_err_key[i], mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (
                            i + 1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


def load_traj_model(data):
    '''
    The trajectory model predicts trajectory that will be use for infilling in motion model.
    Create a trajectory model that produces trajectory to be inptained by the motion model.
    '''
    print("Setting traj model ...")
    # NOTE: Hard-coded trajectory model location
    traj_model_path = "./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt"
    args_traj = eval_args(model_path=traj_model_path)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.num_frames = 196 # This must be 196!

    # print(args_traj.__dict__)
    # print(args_traj.arch)
    traj_model, traj_diffusion = create_model_and_diffusion(args_traj, data)

    print(f"Loading traj model checkpoints from [{args_traj.model_path}]...")
    load_saved_model(traj_model, args_traj.model_path) # , use_avg_model=GEN_USE_TRAJ_AVG_MODEL)

    if args_traj.guidance_param != 1:
        traj_model = ClassifierFreeSampleModel(
            traj_model)  # wrapping model with the classifier-free sampler
    traj_model.to(dist_util.dev())
    traj_model.eval()  # disable random masking
    return traj_model, traj_diffusion


def load_model(args, data):
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
    # Done loading main model
    ###################################
    # Load trajectory model if needed
    if args.gen_two_stages:
        traj_model, traj_diffusion = load_traj_model(data)
    else:
        traj_model, traj_diffusion = None, None

    return model, diffusion, traj_model, traj_diffusion


def load_dataset(args, n_frames, split, hml_mode):
    # (name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt'
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=n_frames,
        split=split,
        hml_mode=hml_mode,
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


if __name__ == '__main__':
    # speed up the eval
    torch.set_num_threads(1)

    # assert flag.TRAIN_MAX_LEN == 196, "UNET length during evaluation works best with 196!"
    
    # args = evaluation_parser()
    args = eval_args()

    # NOTE: test use_ddim
    # args.use_ddim = True

    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.num_frames = 196 # This must be 196!
    args.gen_two_stages = True
    skip_first_stage = False

    if_ddim = "_ddim" if args.use_ddim else ""
    fixseed(args.seed)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    # log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_cond_{}_{}'.format(name, niter))
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_cond_{}_{}_{}'.format(name, niter, if_ddim))

    # save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_cond_{niter}{"_avg" if flag.GEN_USE_AVG_MODEL else ""}')
    save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_cond_{niter}{if_ddim}')
    print('> Saving the generated motion to {}'.format(save_dir))

    # NOTE: Set the final line imputing step t here. This is when we stop imputing with point-to-point trajectory
    # and start imputing with key locations only
    impute_until = 100  # int(args.impute_until) # 100
    print('> Impute the trajectory with point-to-point until t = {}'.format(impute_until))
    # skip_first_stage = True  # bool(args.skip_first) # True
    print('> Skip first stage = {}'.format(skip_first_stage))

    args.eval_mode = 'wo_mm'
    print(f'Eval mode [{args.eval_mode}]')

    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'
    print(f'Will save to log file [{log_file}]')    
    
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        # num_samples_limit = 32
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        # diversity_times = 10
        # replication_times = 2
        replication_times = 20 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        # mm_num_samples = 30
        # mm_num_repeats = 2
        # mm_num_times = 1
        diversity_times = 300
        replication_times = 5 # 1  # about 15 Hrs
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()
    
    logger.log("creating data loader...")
    split = 'test'
    gt_loader = load_dataset(args, args.num_frames, split, hml_mode='gt')
    gen_loader = load_dataset(args, args.num_frames, split, hml_mode='eval')
    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    num_actions = gen_loader.dataset.num_actions

    # logger.log("Creating model and diffusion...")
    # args.predict_xstart = flag.PREDICT_XSTART # True
    # # # Use DDIM for generation
    # # args.use_ddim = flag.GEN_DDIM # True
    # args.abs_3d = flag.ABS_3D # True
    # # args.do_inpaint = flag.GEN_INPAINTING # True
    # # args.traj_only = flag.TRAJECTORY_MODEL # True
    # model, diffusion = create_model_and_diffusion(args, gen_loader)

    # logger.log(f"Loading checkpoints from [{args.model_path}]...")
    # state_dict = torch.load(args.model_path, map_location='cpu')
    # load_model_wo_clip(model, state_dict)

    # if args.guidance_param != 1:
    #     model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    # model.to(dist_util.dev())
    # model.eval()  # disable random masking

    # motion_model, motion_diffusion, traj_model, traj_diffusion = load_2stage_model(args)
    motion_model, motion_diffusion, traj_model, traj_diffusion = load_model(args, gen_loader)
    model_dict = {"motion": motion_model, "traj": traj_model}
    diffusion_dict = {"motion": motion_diffusion, "traj": traj_diffusion}

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald':
        lambda seed: get_mdm_loader_cond(
            model_dict, diffusion_dict, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, num_samples_limit, args.guidance_param,
            seed=seed,
            save_dir=save_dir,
            impute_until=impute_until,
            skip_first_stage=skip_first_stage,
            use_ddim=args.use_ddim,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
