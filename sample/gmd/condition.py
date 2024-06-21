import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def get_target_from_kframes(kframes, batch_size, m_length, device):
    '''Output from this function is already repeated by batch_size and moved to the target device.
    Return:
        target: [bs, m_length, 22, 3]
        target_mask: [bs, m_length, 22, 3]
    '''
    # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
    target = torch.zeros([m_length, 22, 3], device=device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    for kframe in kframes:
        key_step, posi = kframe
        key_step -= 1
        posi = torch.tensor(posi, device=device)
        target[key_step, 0, [0, 2]] = posi
        target_mask[key_step, 0, [0, 2]] = True

    target = target.repeat(batch_size, 1, 1, 1)
    target_mask = target_mask.repeat(batch_size, 1, 1, 1)
    return target, target_mask


def get_target_and_inpt_from_kframes_batch(gt_skel_motions, sampled_keyframes, dataset):
    '''Output from this function is already match the batch_size and moved to the target device.
    Return:
        target: [bs, max_length, 22, 3]
        target_mask: [bs, max_length, 22, 3]
    '''
    batch_size, n_keyframe = sampled_keyframes.shape[:2]
    max_length = gt_skel_motions.shape[-1]
    data_device = gt_skel_motions.device

    target = torch.zeros([batch_size, max_length, 22, 3], device=data_device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    ### For loss target, we only compute with respect to the key points
    # Loop to fill in each entry in the batch f
    for idx in range(batch_size):
        key_posi = gt_skel_motions[idx, 0, :, :].permute(1, 0)  # [max_length, 3]
        for kframe in sampled_keyframes[idx]:
            target[idx, kframe, 0, [0, 2]] = key_posi[kframe, [0, 2]]
            target_mask[idx, kframe, 0, [0, 2]] = True

    ### For inpainting ###
    # inpaint_motion shape [batch, feat_dim, 1, 196], same as model output
    feat_dim = 4  # For trajectory
    inpaint_traj = torch.zeros([batch_size, feat_dim, 1, max_length], device=data_device)
    inpaint_traj_mask = torch.zeros_like(inpaint_traj, dtype=torch.bool)
    # For second stage inpainting, we only use key locations as target instead of the interpolated lines
    inpaint_traj_points = torch.zeros_like(inpaint_traj)
    inpaint_traj_mask_points = torch.zeros_like(inpaint_traj_mask)

    # For motion inpaint (second stage only)
    motion_feat_dim = 263
    inpaint_motion = torch.zeros([batch_size, motion_feat_dim, 1, max_length], device=data_device)
    inpaint_mask = torch.zeros_like(inpaint_motion, dtype=torch.bool)
    # For second stage inpainting, we only use key locations as target instead of the interpolated lines
    inpaint_motion_points = torch.zeros_like(inpaint_motion)
    inpaint_mask_points = torch.zeros_like(inpaint_mask)

    # we draw a point-to-point line between key locations and impute
    INTERPOLATE = True
    for idx in range(batch_size):
        key_posi = gt_skel_motions[idx, 0, :, :].permute(1, 0)  # [max_length, 3]
        # Initialization
        cur_x, cur_z = 0.0, 0.0
        last_kframe = 0
        # Each key frame in batch
        for kframe_id, kframe_t in enumerate(sampled_keyframes[idx]):

            diff = kframe_t - last_kframe
            # Get (x,z) from the key locations
            (xx, zz) = key_posi[kframe_t, [0, 2]]
            if INTERPOLATE:
                # Loop to get an evenly space trajectory
                for i in range(diff):
                    inpaint_traj[idx, 1, 0, last_kframe + i] = (cur_x + (xx-cur_x) * i / diff)
                    inpaint_traj[idx, 2, 0, last_kframe + i] = (cur_z + (zz-cur_z) * i / diff)
                    inpaint_traj_mask[idx, [1,2], 0, last_kframe + i] = True
            else:
                # No loop
                inpaint_traj[idx, 1, 0, kframe_t] = xx
                inpaint_traj[idx, 2, 0, kframe_t] = zz
                inpaint_traj_mask[idx, [1, 2], 0, kframe_t] = True

            inpaint_traj_points[idx, 1, 0, kframe_t] = xx
            inpaint_traj_points[idx, 2, 0, kframe_t] = zz
            inpaint_traj_mask_points[idx, [1, 2], 0, kframe_t] = True

            cur_x, cur_z = xx, zz
            last_kframe = kframe_t
            # Add last key point
            if kframe_id == len(sampled_keyframes[idx]) - 1:
                # print("add", frame)
                inpaint_traj[idx, 1, 0, kframe_t] = xx
                inpaint_traj[idx, 2, 0, kframe_t] = zz
                inpaint_traj_mask[idx, [1, 2], 0, kframe_t] = True

    # import pdb; pdb.set_trace()
    # plt.scatter(inpaint_traj[0, 1, 0, :].detach().numpy(), inpaint_traj[0, 2, 0, :].detach().numpy())
    # Transform the inpaint_traj to the model input space

    # Copy the traj values into inpainted motion
    # For motion we do not have to do transform
    inpaint_motion[:, :4, :, :] = inpaint_traj[:, :4, :, :]
    inpaint_motion_points[:, :4, :, :] = inpaint_traj_points[:, :4, :, :]
    inpaint_mask[:, :4, :, :] = inpaint_traj_mask[:, :4, :, :]
    inpaint_mask_points[:, :4, :, :] = inpaint_traj_mask_points[:, :4, :, :]


    # No random projection for the first stage for all the case
    # [bs, 4, 1, 196]
    inpaint_traj = dataset.t2m_dataset.transform_th(inpaint_traj.permute(0, 2, 3, 1), traject_only=True,
                                                      use_rand_proj=False).permute(0, 3, 1, 2)
    # [bs, 4, 1, 196]
    inpaint_traj_points = dataset.t2m_dataset.transform_th(inpaint_traj_points.permute(0, 2, 3, 1), traject_only=True,
                                                      use_rand_proj=False).permute(0, 3, 1, 2)
    return (target, target_mask, inpaint_traj, inpaint_traj_mask, inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion, inpaint_mask, inpaint_motion_points, inpaint_mask_points)


# Our version for arbitrary keyframes
def compute_kps_error_arbitrary(cur_motion, gt_skel_motions, sampled_keyframes, traj_only=True):
    ''' Compute root position error over keyframes
    cur_motion [32, 22, 3, 196]
    gt_skel_motions [32, 22, 3, 196]
    sampled_keyframes 32 lists of int
    '''
    # Pad the sampled_keyframes to the same length
    sampled_keyframes = [torch.tensor(ks) for ks in sampled_keyframes]
    from torch.nn.utils.rnn import pad_sequence
    sampled_keyframes = pad_sequence(sampled_keyframes, batch_first=True, padding_value=-1)

    batch_size = cur_motion.shape[0]
    dist_err = torch.zeros_like(sampled_keyframes, dtype=torch.float)
    sampled_keyframes = sampled_keyframes.long()

    for ii in range(batch_size):
        cur_keyframes = sampled_keyframes[ii]
        if traj_only:
            # root position only
            motion_xz = cur_motion[ii, 0:1, [0, 2]]
            gt_xz = gt_skel_motions[ii, 0:1, [0, 2]]  # [1, 2, 196]
        else:
            # full keyframe
            motion_xz = cur_motion[ii]
            gt_xz = gt_skel_motions[ii] # [22, 3, 196]
        # This should be location error for each key frames in meter
        cur_err = torch.linalg.norm(motion_xz[..., cur_keyframes] - gt_xz[..., cur_keyframes], dim=1).mean(dim=0) #FIXME: May need to cut the error to the length of the keyframes
        dist_err[ii, :cur_err.shape[0]] = cur_err
    return dist_err


def compute_kps_error(cur_motion, gt_skel_motions, sampled_keyframes):
    '''
    cur_motion [32, 22, 3, 196]
    gt_skel_motions [32, 22, 3, 196]
    sampled_keyframes [32, 5]
    '''
    batch_size = cur_motion.shape[0]
    dist_err = torch.zeros_like(sampled_keyframes, dtype=torch.float)
    sampled_keyframes = sampled_keyframes.long()
    for ii in range(batch_size):
        cur_keyframes = sampled_keyframes[ii]
        motion_xz = cur_motion[ii, 0, [0, 2]]
        gt_xz = gt_skel_motions[ii, 0, [0, 2]]  # [2, 196]
        # This should be location error for each key frames in meter
        cur_err = torch.linalg.norm(motion_xz[:, cur_keyframes] - gt_xz[:, cur_keyframes], dim=0)
        dist_err[ii, :] = cur_err[:]
    return dist_err


def get_inpainting_motion(kframes,
                          batch_size,
                          m_length,
                          abs_3d=False,
                          traj_only=False,
                          transform_fn=None,
                          gen_two_stage=False,
                          use_rand_proj=True):
    # x shape [1, 263, 1, 120]
    # import pdb; pdb.set_trace()
    # m_length = int(flag.GEN_MOTION_LENGTH_CUT * 20)
    if traj_only:
        feat_dim = 4
    else:
        feat_dim = 263
    INTERPOLATE = False
    if abs_3d:
        mask = torch.zeros([feat_dim, m_length], dtype=torch.bool)
        # mask[[1,2], :] = True

        cur_x, cur_z = 0.0, 0.0
        last_kframe = 0
        motion = torch.zeros([feat_dim, m_length])
        init_motion = torch.zeros([feat_dim, m_length])

        for idx, (frame, posi) in enumerate(kframes):
            # Offset as the first frame is frame_0
            frame = frame - 1
            (xx, zz) = posi
            diff = frame - last_kframe
            # print("key frame", frame)
            for i in range(diff):
                if INTERPOLATE:
                    motion[1, last_kframe + i] = (cur_x + (xx-cur_x) * i / diff)
                    motion[2, last_kframe + i] = (cur_z + (zz-cur_z) * i / diff)
                    # if True: # i % 5 == 0:
                    mask[[1,2], last_kframe + i] = True
                # print("add", last_kframe + i)
                init_motion[1, last_kframe + i] = (cur_x + (xx-cur_x) * i / diff)
                init_motion[2, last_kframe + i] = (cur_z + (zz-cur_z) * i / diff)
            if not INTERPOLATE:
                motion[1, frame] = xx
                motion[2, frame] = zz
                mask[[1, 2], frame] = True
            cur_x, cur_z = xx, zz
            last_kframe = frame
            if idx == len(kframes) - 1:
                # print("add", frame)
                init_motion[1, frame] = xx
                init_motion[2, frame] = zz

    else:
        mask = torch.zeros([feat_dim, m_length], dtype=torch.bool)
        motion = torch.zeros([feat_dim, m_length])
        init_motion = torch.zeros([feat_dim, m_length])

    # import pdb; pdb.set_trace()
    if gen_two_stage:
        motion = transform_fn(motion.permute(1, 0), traject_only=True, use_rand_proj=False).permute(1, 0)
        init_motion = transform_fn(init_motion.permute(1, 0), traject_only=True, use_rand_proj=False).permute(1, 0)
        init_motion[[0, 3], :] = 0.0
    elif not use_rand_proj:
        # If we do not use random projection, we can transform the motion here
        # without passing the transform function into the model.
        motion = transform_fn(motion.permute(1, 0), traject_only=traj_only).permute(1, 0)
        init_motion = transform_fn(init_motion.permute(1, 0), traject_only=traj_only).permute(1, 0)
        init_motion[[0, 3], :] = 0.0

    motion = motion.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)
    mask = mask.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)
    init_motion = init_motion.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)

    # Return shape [batch, feat_dim, 1, 120]
    return motion, mask, init_motion


def get_inpainting_motion_from_gt(gt_skel_motions, batch_size, model_device, lengths, inv_transform_fn=None):
    max_length = gt_skel_motions.shape[-1]
    data_device = gt_skel_motions.device
    feat_dim = 263
    # gt_skel_motions [32, 22, 3, 196]
    # target: [bs, max_length, 22, 3]
    # target_mask: [bs, max_length, 22, 3]

    # We transform the motion back to the real motion space
    # because the model with projection can only do inpainting in the original motion space at test time
    motion = torch.zeros([batch_size, feat_dim, 1, max_length], device=gt_skel_motions.device)
    motion[:, [1,2], 0, :] = gt_skel_motions.permute(0, 2, 1, 3)[:, [0,2], 0, :]

    target = torch.zeros([batch_size, max_length, 22, 3], device=gt_skel_motions.device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    # import pdb; pdb.set_trace()
    target[:, :, 0, [0, 2]] = motion.permute(0, 3, 2, 1)[:, :, 0,[1, 2]]
    target_mask[:, :, 0, [0, 2]] = True

    for i, motion_len in enumerate(lengths):
        # Duplicate the last position in the ground truth until the max_length
        # This is to avoid inpainting with the wrong values
        motion[i, [1,2], :, motion_len:] = motion[i, [1,2], :, motion_len-1, None]
        target_mask[i, motion_len:, 0, [0, 2]] = False

    mask = torch.zeros_like(motion, dtype=torch.bool)
    # Impute (x,z)
    mask[:, [1, 2], :, :] = True

    motion = motion.to(model_device)
    mask = mask.to(model_device)

    target = target.to(model_device)
    target_mask = target_mask.to(model_device)

    return motion, mask, target, target_mask


def get_inpainting_motion_from_traj(traj_out, inv_transform_fn=None, max_impute=None):
    feat_dim = 263
    m_length = traj_out.shape[-1]
    batch_size = traj_out.shape[0]
    # We transform the motion back to the real motion space
    # because the model with projection can only do inpainting in the original motion space at test time
    motion = torch.zeros([batch_size, feat_dim, 1, m_length], device=traj_out.device)
    traj_ori_space = inv_transform_fn(traj_out.permute(0, 2, 3, 1),
                                      traject_only=True,
                                      use_rand_proj=False).permute(
                                          0, 3, 1, 2)  # [1, feat_dim, 1, 120]
    motion[:, [1, 2], :, :] = traj_ori_space[:, [1, 2], :, :]
    mask = torch.zeros_like(motion, dtype=torch.bool)
    # Impute (x,z)
    if max_impute is not None:
        mask[:, [1, 2], :, :max_impute] = True
    else:
        mask[:, [1, 2], :, :] = True
    # import pdb; pdb.set_trace()
    # maybe we should only set the mask until 120 ?

    return motion, mask


### Condition on key location
def cond_fn_key_location(
    x,
    t,
    p_mean_var,
    y=None,
    kframes=None,
    target=None,
    target_mask=None,
    transform=None,
    inv_transform=None,
    abs_3d=False,
    classifiler_scale=10.0,
    out_list=None,
    reward_model=None,
    reward_model_args=None,
    use_mse_loss=False,
    guidance_style='xstart',
    stop_cond_from=0,
    use_rand_projection=False,
    motion_length_cut=6.0,
    # diffusion=None,
):
    """
    Args:
        target: [bs, 120, 22, 3]
        target_mask: [bs, 120, 22, 3]
    """
    # Stop condition
    if int(t[0]) < stop_cond_from:
        return torch.zeros_like(x)
    # x_t_previous = p_mean_var['mean']
    # assert y is not None
    # x shape [1, 263, 1, 120]
    with torch.enable_grad():
        # x_in = x.detach().requires_grad_(True)
        # reward_model = None

        # NOTE: optimizing to the whole trajectory is not good.
        gt_style = 'target'

        if gt_style == 'target':
            if guidance_style == 'xstart':
                if reward_model is not None:
                    # If the reward model is provided, we will use xstart from the reward model instead.
                    # The reward model predict M(x_start | x_t)
                    # The gradient is always computed w.r.t. x_t
                    x = x.detach().requires_grad_(True)
                    reward_model_output = reward_model(
                        x, t, **reward_model_args)  # this produces xstart
                    xstart_in = reward_model_output
                else:
                    xstart_in = p_mean_var['pred_xstart']
            elif guidance_style == 'eps':
                # using epsilon style guidance
                assert reward_model is None, "there is no need for the reward model in this case"
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            n_joints = 22
            # out_path = y['log_name']
            # (pose,x,z,y)
            if y['traj_model']:
                use_rand_proj = False
            else:
                use_rand_proj = use_rand_projection
            x_in_pose_space = inv_transform(
                xstart_in.permute(0, 2, 3, 1),
                traject_only=y['traj_model'],
                use_rand_proj=use_rand_proj
            )  # [bs, 1, 120, 263]
            # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]

            # (x,y,z)
            # [bs, 1, 120, njoints=22, nfeat=3]
            x_in_joints = recover_from_ric(x_in_pose_space,
                                        n_joints,
                                        abs_3d=abs_3d)
            # plt.scatter(x_in_joints[0,0, :, 0, 0].detach().cpu().numpy(), x_in_joints[0,0, :, 0, 2].detach().cpu().numpy())

            # trajectory is the first (pelvis) joint
            # [bs, 120, 3]
            trajec = x_in_joints[:, 0, :, 0, :]
            # plt.scatter(trajec[:,0].detach().cpu().numpy(), trajec[:,2].detach().cpu().numpy())
            # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
            # Only care about XZ position for now. Y-axis is going up from the ground
            batch_size = trajec.shape[0]
            # if flag.MODEL_NAME[1] == 'unet':
            cut_frame = int(motion_length_cut * 20)
            trajec = trajec[:, :cut_frame, :]

            loss_mask_type = 'new'
            if loss_mask_type == 'legacy':
                # the old loss function is not correct in terms of masking
                if use_mse_loss:
                    loss_sum = F.mse_loss(trajec * target_mask[:, :cut_frame, 0, :], target[:, :cut_frame, 0, :],
                                        reduction='sum')
                else:
                    loss_sum = F.l1_loss(trajec * target_mask[:, :cut_frame, 0, :], target[:, :cut_frame, 0, :],
                                        reduction='sum')
            elif loss_mask_type == 'new':
                # fixed the masking problem
                if use_mse_loss:
                    loss_sum = F.mse_loss(trajec , target[:, :cut_frame, 0, :],
                                        reduction='none') * target_mask[:, :cut_frame, 0, :]
                else:
                    loss_sum = F.l1_loss(trajec , target[:, :cut_frame, 0, :],
                                        reduction='none') * target_mask[:, :cut_frame, 0, :]
                loss_sum = loss_sum.sum()
            else:
                raise NotImplementedError()

            # Scale the loss up so that we get the same gradient as if each sample is computed individually
            loss_sum = loss_sum / target_mask.sum() * batch_size
        elif gt_style == 'inpainting_motion':
            batch_size = x.shape[0]
            # [bs, 4, 1, 120]
            xstart_in = p_mean_var['pred_xstart']
            inpainted_motion = y['current_inpainted_motion']
            inpainting_mask = y['current_inpainting_mask']
            # Inpainting motion
            if use_mse_loss:
                loss_sum = F.mse_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
            else:
                loss_sum = F.l1_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
            # Scale the loss up so that we get the same gradient as if each sample is computed individually
            loss_sum = loss_sum.sum() / inpainting_mask.sum() * batch_size
            print('loss:', float(loss_sum), 'count:', int(inpainting_mask.sum()))
        else:
            raise NotImplementedError()

        if int(t[0]) % 100 == 0:
            print("%03d: %f" % (int(t[0]), float(loss_sum) / batch_size))

        grad = torch.autograd.grad(-loss_sum, x)[0]  # retain_graph=True
        # print("grad_norm", grad.norm())
        return grad * classifiler_scale  # * (1000. - t) / 100.0


class CondKeyLocations:
    def __init__(self,
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 classifiler_scale=10.0,
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 motion_length_cut=6.0,
                 print_every=None,
                 ):
        self.target = target
        self.target_mask = target_mask
        self.transform = transform
        self.inv_transform = inv_transform
        self.abs_3d = abs_3d
        self.classifiler_scale = classifiler_scale
        self.reward_model = reward_model
        self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.use_rand_projection = use_rand_projection
        self.motion_length_cut = motion_length_cut
        self.cut_frame = int(self.motion_length_cut * 20)
        self.print_every = print_every

        self.n_joints = 22
        # NOTE: optimizing to the whole trajectory is not good.
        self.gt_style = 'target'  # 'inpainting_motion'

    def __call__(self, x, t, p_mean_var, y=None,): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        # Stop condition
        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)
        assert y is not None
        # x shape [bs, 263, 1, 120]
        # import pdb; pdb.set_trace()
        with torch.enable_grad():
            if self.gt_style == 'target':
                if self.guidance_style == 'xstart':
                    if self.reward_model is not None:
                        # If the reward model is provided, we will use xstart from the
                        # reward model instead.The reward model predict M(x_start | x_t).
                        # The gradient is always computed w.r.t. x_t
                        x = x.detach().requires_grad_(True)
                        reward_model_output = self.reward_model(
                            x, t, **self.reward_model_args)  # this produces xstart
                        xstart_in = reward_model_output
                    else:
                        xstart_in = p_mean_var['pred_xstart']
                elif self.guidance_style == 'eps':
                    # using epsilon style guidance
                    assert self.reward_model is None, "there is no need for the reward model in this case"
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
                if y['traj_model']:
                    use_rand_proj = False  # x contains only (pose,x,z,y)
                else:
                    use_rand_proj = self.use_rand_projection
                x_in_pose_space = self.inv_transform(
                    xstart_in.permute(0, 2, 3, 1),
                    traject_only=y['traj_model'],
                    use_rand_proj=use_rand_proj
                )  # [bs, 1, 120, 263]
                # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]
                # Compute (x,y,z) shape [bs, 1, 120, njoints=22, nfeat=3]
                x_in_joints = recover_from_ric(x_in_pose_space, self.n_joints,
                                            abs_3d=self.abs_3d)
                # trajectory is the first joint (pelvis) shape [bs, 120, 3]
                trajec = x_in_joints[:, 0, :, 0, :]
                # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
                # Only care about XZ position for now. Y-axis is going up from the ground
                batch_size = trajec.shape[0]
                trajec = trajec[:, :self.cut_frame, :]

                if self.use_mse_loss:
                    loss_sum = F.mse_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                        reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
                else:
                    loss_sum = F.l1_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                        reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
                loss_sum = loss_sum.sum()
                # Scale the loss up so that we get the same gradient as if each sample is computed individually
                loss_sum = loss_sum / self.target_mask.sum() * batch_size

            elif self.gt_style == 'inpainting_motion':
                batch_size = x.shape[0]
                # [bs, 4, 1, 120]
                xstart_in = p_mean_var['pred_xstart']
                inpainted_motion = y['current_inpainted_motion']
                inpainting_mask = y['current_inpainting_mask']
                # Inpainting motion
                if self.use_mse_loss:
                    loss_sum = F.mse_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
                else:
                    loss_sum = F.l1_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
                # Scale the loss up so that we get the same gradient as if each sample is computed individually
                loss_sum = loss_sum.sum() / inpainting_mask.sum() * batch_size
                print('loss:', float(loss_sum), 'count:', int(inpainting_mask.sum()))

            else:
                raise NotImplementedError()

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: %f" % (int(t[0]), float(loss_sum) / batch_size))

            grad = torch.autograd.grad(-loss_sum, x)[0]
            return grad * self.classifiler_scale


class CondKeyLocationsWithSdf:
    def __init__(self,
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 classifiler_scale=10.0,
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 motion_length_cut=6.0,
                 obs_list=[],
                 print_every=None,
                 w_colli=5.0,
                 ):
        self.target = target
        self.target_mask = target_mask
        self.transform = transform
        self.inv_transform = inv_transform
        self.abs_3d = abs_3d
        self.classifiler_scale = classifiler_scale
        self.reward_model = reward_model
        self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.use_rand_projection = use_rand_projection
        self.motion_length_cut = motion_length_cut
        self.cut_frame = int(self.motion_length_cut * 20)
        self.obs_list = obs_list
        self.print_every = print_every

        self.n_joints = 22
        self.w_colli = w_colli
        self.use_smooth_loss = False

    def __call__(self, x, t, p_mean_var, y=None,): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        # Stop condition
        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)
        assert y is not None
        # x shape [bs, 263, 1, 120]
        with torch.enable_grad():
            if self.guidance_style == 'xstart':
                if self.reward_model is not None:
                    # If the reward model is provided, we will use xstart from the
                    # reward model instead.The reward model predict M(x_start | x_t).
                    # The gradient is always computed w.r.t. x_t
                    x = x.detach().requires_grad_(True)
                    reward_model_output = self.reward_model(
                        x, t, **self.reward_model_args)  # this produces xstart
                    xstart_in = reward_model_output
                else:
                    xstart_in = p_mean_var['pred_xstart']
            elif self.guidance_style == 'eps':
                # using epsilon style guidance
                assert self.reward_model is None, "there is no need for the reward model in this case"
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            if y['traj_model']:
                use_rand_proj = False  # x contains only (pose,x,z,y)
            else:
                use_rand_proj = self.use_rand_projection
            x_in_pose_space = self.inv_transform(
                xstart_in.permute(0, 2, 3, 1),
                traject_only=y['traj_model'],
                use_rand_proj=use_rand_proj
            )  # [bs, 1, 120, 263]
            # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]
            # Compute (x,y,z) shape [bs, 1, 120, njoints=22, nfeat=3]
            x_in_joints = recover_from_ric(x_in_pose_space, self.n_joints,
                                        abs_3d=self.abs_3d)
            # trajectory is the first joint (pelvis) shape [bs, 120, 3]
            trajec = x_in_joints[:, 0, :, 0, :]
            # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
            # Only care about XZ position for now. Y-axis is going up from the ground
            batch_size = trajec.shape[0]
            trajec = trajec[:, :self.cut_frame, :]
            if self.use_mse_loss:
                loss_kps = F.mse_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                    reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
            else:
                loss_kps = F.l1_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                    reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
            loss_kps = loss_kps.sum()

            loss_colli = 0.0
            for ((c_x, c_z), rad) in self.obs_list:
                cent = torch.tensor([c_x, c_z], device=trajec.device)
                dist = torch.norm(trajec[:, :, [0, 2]] - cent, dim=2)
                dist = torch.clamp(rad - dist, min=0.0)
                loss_colli += dist.sum() / trajec.shape[1] * self.w_colli

            if self.use_smooth_loss:
                loss_smooth_traj = F.mse_loss(trajec[:, 1:, [0, 2]], trajec[:, :-1, [0, 2]])
                loss_sum += loss_smooth_traj

            loss_kps = loss_kps / self.target_mask.sum() * batch_size
            loss_kps = loss_kps
            loss_sum = loss_kps + loss_colli

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: %f, %f" % (int(t[0]), float(loss_kps), float(loss_colli)))

            grad = torch.autograd.grad(-loss_sum, x)[0]
            return grad * self.classifiler_scale


### Condition on key location
def cond_fn_sdf(
    x,
    t,
    p_mean_var,
    y=None,
    kframes=None,
    target=None,
    target_mask=None,
    transform=None,
    inv_transform=None,
    abs_3d=False,
    classifiler_scale=10.0,
    out_list=None,
    reward_model=None,
    reward_model_args=None,
    use_mse_loss=False,
    guidance_style='xstart',
    stop_cond_from=0,
    use_rand_projection=False,
    motion_length_cut=6.0,
    # diffusion=None,
    obs_list=[], # Each obstacle is a pair of [(x,z) center, radius]
):
    """
    Args:
        target: [bs, 120, 22, 3]
        target_mask: [bs, 120, 22, 3]
    """
    # Stop condition
    if int(t[0]) < stop_cond_from:
        return torch.zeros_like(x)
    # x_t_previous = p_mean_var['mean']
    # assert y is not None
    # x shape [1, 263, 1, 120]
    with torch.enable_grad():
        # x_in = x.detach().requires_grad_(True)
        # reward_model = None
        if guidance_style == 'xstart':
            if reward_model is not None:
                # If the reward model is provided, we will use xstart from the reward model instead.
                # The reward model predict M(x_start | x_t)
                # The gradient is always computed w.r.t. x_t
                x = x.detach().requires_grad_(True)
                reward_model_output = reward_model(
                    x, t, **reward_model_args)  # this produces xstart
                xstart_in = reward_model_output
            else:
                xstart_in = p_mean_var['pred_xstart']
        elif guidance_style == 'eps':
            # using epsilon style guidance
            assert reward_model is None, "there is no need for the reward model in this case"
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        n_joints = 22
        # out_path = y['log_name']
        # (pose,x,z,y)
        if y['traj_model']:
            use_rand_proj = False
        else:
            use_rand_proj = use_rand_projection
        x_in_pose_space = inv_transform(
            xstart_in.permute(0, 2, 3, 1),
            traject_only=y['traj_model'],
            use_rand_proj=use_rand_proj
        )  # [bs, 1, 120, 263]
        # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]

        # (x,y,z)
        # [bs, 1, 120, njoints=22, nfeat=3]
        x_in_joints = recover_from_ric(x_in_pose_space,
                                       n_joints,
                                       abs_3d=abs_3d)
        # plt.scatter(x_in_joints[0,0, :, 0, 0].detach().cpu().numpy(), x_in_joints[0,0, :, 0, 2].detach().cpu().numpy())

        # trajectory is the first (pelvis) joint
        # [bs, 120, 3]
        trajec = x_in_joints[:, 0, :, 0, :]
        # plt.scatter(trajec[:,0].detach().cpu().numpy(), trajec[:,2].detach().cpu().numpy())
        # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
        # Only care about XZ position for now. Y-axis is going up from the ground
        batch_size = trajec.shape[0]
        # if flag.MODEL_NAME[1] == 'unet':
        cut_frame = int(motion_length_cut * 20)
        trajec = trajec[:, :cut_frame, :]

        if use_mse_loss:
            loss_kps = F.mse_loss(trajec * target_mask[:, :cut_frame, 0, :], target[:, :cut_frame, 0, :],
                                  reduction='sum')
        else:
            loss_kps = F.l1_loss(trajec * target_mask[:, :cut_frame, 0, :], target[:, :cut_frame, 0, :],
                                 reduction='sum')
        # import pdb; pdb.set_trace()
        loss_colli = 0.0
        w_colli = 5.0

        for ((c_x, c_z), rad) in obs_list:
            cent = torch.tensor([c_x, c_z], device=trajec.device)
            dist = torch.norm(trajec[:, :, [0, 2]] - cent, dim=2)
            dist = torch.clamp(rad - dist, min=0.0)
            loss_colli += dist.sum() / trajec.shape[1] * w_colli

        use_smooth_loss = False # True
        if use_smooth_loss:
            # reg_smooth = 1e3
            loss_smooth_traj = F.mse_loss(trajec[:, 1:, [0, 2]], trajec[:, :-1, [0, 2]])
            loss_sum += loss_smooth_traj

        w_kps = 1.0 # 0.0
        loss_kps = loss_kps / target_mask.sum() * batch_size
        loss_kps = loss_kps * w_kps
        # Scale the loss up so that we get the same gradient as if each sample is computed individually
        loss_sum = loss_kps + loss_colli

        if int(t[0]) % 100 == 0:
            # import pdb; pdb.set_trace()
            # plt.scatter(trajec[0, :, 0].detach().cpu().numpy(), trajec[0, :, 2].detach().cpu().numpy())
            # print("%03d: %f, %f" % (int(t[0]), float(loss_sum) / batch_size))
            print("%03d: %f, %f" % (int(t[0]), float(loss_kps), float(loss_colli)))

        grad = torch.autograd.grad(-loss_sum, x)[0]  # retain_graph=True
        # print("grad_norm", grad.norm())
        return grad * classifiler_scale



def log_trajectory_from_xstart(xstart,
                               out_path,
                               out_list,
                               t,
                               log_id,
                               kframes=[],
                               inv_transform=None,
                               abs_3d=True,
                               use_rand_proj=False,
                               traject_only=False,
                               n_frames=int(6.0 * 20),
                               combine_to_video=False,
                               obs_list=[]):
    if inv_transform is None:
        return
    x_in_pose_space = inv_transform(
        xstart.permute(0, 2, 3, 1),
        use_rand_proj=use_rand_proj,
        traject_only=traject_only)  # [bs, 1, 120, 263]
    # (x,y,z)
    n_joints = 22
    x_in_joints = recover_from_ric(x_in_pose_space, n_joints,
                                   abs_3d=abs_3d)  # [bs, 1, 120, 22, 3]
    trajec = x_in_joints[:, 0, :, 0, :].detach().cpu().numpy()  # [bs, 120, 3]
    trajec = trajec[:, :n_frames, :]
    prefix = "motion_" if not traject_only else ""

    if len(kframes) == 0:
        kframes = [(0, (0., 0.))]

    for sample_i in range(len(trajec)):
        f, ax = plt.subplots()
        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-3.0, 6.0)
        ax.set_aspect('equal', adjustable='box')
        # ax.axis('off')
        kframe_locs = np.array([b for (a, b) in kframes])
        kframe_steps = np.array([a for (a, b) in kframes]) ## NOTE: <<-- no "-1" here
        ax.scatter(kframe_locs[:, 0], kframe_locs[:, 1], color='r')
        ax.scatter(trajec[sample_i, :, 0], trajec[sample_i, :, 2])
        # Key locations
        ax.scatter(trajec[sample_i, kframe_steps, 0],
                   trajec[sample_i, kframe_steps, 2],
                   color='yellow')
        # Plot location to aviod (if any)
        for ((c_x, c_z), r) in obs_list:
            circ = Circle((c_x, c_z), r, facecolor='red', edgecolor='red', lw=1) # zorder=10
            ax.add_patch(circ)

        ax.title.set_text("rep_%d: denoise step %04d" %
                          (sample_i, 999 - int(t[0])))
        ax.title.set_text("t = %d" %
                          (int(t[0])))
        plt.gca().invert_yaxis()
        plt.savefig(
            os.path.join(
                out_path, "%strajec_" % (prefix) + "%d_%04d.png" %
                (sample_i, 999 - int(t[0])))
                # , bbox_inches='tight'
        )
        # plt.show()
        plt.close()

    if combine_to_video and int(t[0]) == 0:
        for sample_i in range(len(trajec)):
            rep_files = []
            for ss in out_list:
                rep_files.append(
                    os.path.join(out_path, "%strajec_" % (prefix) + "%d_%04d.png" % (sample_i, 999 - ss)))
            rep_pattern = '"' + os.path.join(out_path, "%strajec_" % (prefix) + "%d_*.png" % (sample_i)) + '"'
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            trajec_ani_path = os.path.join(out_path, "%strajec_%d.mp4" % (prefix, sample_i))

            ffmpeg_rep_cmd = (
                f'ffmpeg -y -loglevel warning -framerate 4 -pattern_type glob -i' +
                f' {rep_pattern}'
                f' -c:v libx264 -r 30 -pix_fmt yuv420p' + f' {trajec_ani_path}')
            # import pdb; pdb.set_trace()
            os.system(ffmpeg_rep_cmd)


def opt_inner_loop_v2(x_in, transform, inv_transform, kframes, abs_3d=False):
    ''' These functions are for optimizing x_t at each denoising step, instead of only providing gradient once.
    Currently not in use.
    '''
    def loss_fn(trajec):
        loss_sum = 0.0
        for kframe in kframes:
            time_step, posi = kframe
            # Adjust by one so it is easier to enter kframes
            time_step -= 1
            posi = torch.tensor(posi, device=trajec.device)
            loss_sum += F.mse_loss(trajec[time_step, [0, 2]], posi)
        return loss_sum

    # optimize wrt 'x' (model input space) or 'z' (human motion space)
    OPT_WRT = 'z'

    x_in = x_in.detach().clone()
    x_in_clone = x_in.detach().clone()

    if OPT_WRT == 'x':
        # optimize wrt. input vector (could be projected)
        x_in.requires_grad = True
        opt_vec = [x_in]
    elif OPT_WRT == 'z':
        # optimize wrt. human motion vector
        # z = motion vector representation
        z = inv_transform(x_in.permute(0, 2, 3, 1))
        z.requires_grad = True
        trajec_start = recover_from_ric(z, 22, abs_3d=abs_3d)[0, 0, :,
                                                              0, :]  # [120, 3]
        opt_vec = [z]
    else:
        raise NotImplementedError()

    # opt = optim.SGD(opt_vec, lr=1e-4)
    opt = optim.SGD(opt_vec, lr=5e-2)
    # opt = optim.Adam(opt_vec, lr=1e-2)

    for j in range(200):
        if OPT_WRT == 'x':
            # while changing x, you'll need to refresh the values of the motion vector
            z = inv_transform(x_in.permute(0, 2, 3, 1))  # [1, 1, 120, 263]

        trajec = recover_from_ric(z, 22, abs_3d=abs_3d)[0, 0, :,
                                                        0, :]  # [120, 3]
        loss_traj = loss_fn(trajec)

        # import pdb; pdb.set_trace()
        loss_smooth_traj = F.mse_loss(trajec[1:], trajec[:-1])
        reg_smooth = 1e3

        reg = 1e2
        if OPT_WRT == 'x':
            loss_reg = F.mse_loss(x_in, x_in_clone)
        elif OPT_WRT == 'z':
            loss_reg = F.mse_loss(
                z, inv_transform(x_in_clone.permute(0, 2, 3, 1)))
        loss_sum = loss_traj + reg * loss_reg + reg_smooth * loss_smooth_traj

        if j == 0:
            print('loss traj', float(loss_traj))
            print('loss smooth traj', float(loss_smooth_traj))

        opt.zero_grad()
        loss_sum.backward()

        GRAD_NORM = True
        if GRAD_NORM:
            if OPT_WRT == 'z':
                # gradient normalization
                z.grad = F.normalize(z.grad, dim=3)
            elif OPT_WRT == 'x':
                raise NotImplementedError()

        # don't need it for now
        if float(loss_sum < 0.01):
            break

        opt.step()

    if OPT_WRT == 'z':
        # transform the updated z back to x_in to get the output
        x_in = transform(z).permute(0, 3, 1, 2)
    elif OPT_WRT == 'x':
        # the last x_in is the output
        pass

    # import pdb; pdb.set_trace()
    print('after opt loss traj', float(loss_traj))
    print('after opt loss smooth traj', float(loss_smooth_traj))
    print("grad size", float(torch.norm(x_in - x_in_clone)))

    VIS_TRAJ = False  # True
    if VIS_TRAJ:
        with torch.no_grad():
            plt.plot(trajec_start[:, 0].detach().cpu(),
                     trajec_start[:, 2].detach().cpu())
            plt.title("After - Red")

            # x_new = x_in_clone + cum_grad
            x_new_adjust = inv_transform(x_in.permute(0, 2, 3, 1))
            trajec_new = recover_from_ric(x_new_adjust, 22,
                                          abs_3d=abs_3d)[0, 0, :,
                                                         0, :]  # [120, 3]
            plt.plot(trajec_new[:, 0].detach().cpu(),
                     trajec_new[:, 2].detach().cpu(),
                     color='r')
            c = 1.1
            plt.xlim(-c, 3 + c)
            plt.ylim(-c, 3 + c)
            plt.show()

    # x_in is the output
    return x_in.detach()


def opt_inner_loop(x_in, transform, inv_transform, kframes, abs_3d=False):
    '''
    These functions are for optimizing x_t at each denoising step, instead of only providing gradient once.
    Currently not in use.

    Compute the desired x_0 based on the objective function. Return new x_0.
    x_in: x_0 at time t predicted by the model.
    '''
    def loss_fn(trajec):
        loss_sum = 0.0
        for kframe in kframes:
            time_step, posi = kframe
            # Adjust by one so it is easier to enter kframes
            time_step -= 1
            posi = torch.tensor(posi, device=trajec.device)
            loss_sum += F.mse_loss(trajec[time_step, [0, 2]], posi)

    #         loss_sum += torch.norm(trajec[time_step, [0,2]] - posi)
        return loss_sum

    x_in = x_in.detach().clone()
    x_in_clone = x_in.detach().clone()
    x_in.requires_grad = True
    # trajec.requires_grad = True
    # opt = optim.SGD([x_in], lr=1e-4)
    # opt = optim.SGD([x_in], lr=1e-1)
    opt = optim.Adam([x_in], lr=5e-3)

    x_in_adjust = inv_transform(x_in.permute(0, 2, 3, 1))
    trajec_start = recover_from_ric(x_in_adjust, 22,
                                    abs_3d=abs_3d)[0, 0, :, 0, :]  # [120, 3]

    # cum_grad = torch.zeros_like(x_in)
    zeros = torch.zeros_like(x_in.permute(0, 2, 3, 1))

    for j in range(200):
        x_in_adjust = inv_transform(x_in.permute(0, 2, 3,
                                                 1))  # [1, 1, 120, 263]
        trajec = recover_from_ric(x_in_adjust, 22,
                                  abs_3d=abs_3d)[0, 0, :, 0, :]  # [120, 3]

        # Try to minimize turning
        # import pdb; pdb.set_trace()
        # loss_turning = F.mse_loss(x_in_adjust[:, :, :, 0], zeros[:, :, :, 0])
        # reg_turning = 1e2

        loss_smooth_traj = F.mse_loss(trajec[1:], trajec[:-1])
        reg_smooth = 1e1

        # loss_turn_smooth = F.mse_loss(x_in_adjust[:, :, :-1, 0], x_in_adjust[:, :, 1:, 0])
        # reg_turn_smooth = 1e1

        # loss_smooth_xy = F.mse_loss(x_in_adjust[:, :, :-1, [1, 2]], x_in_adjust[:, :, 1:, [1, 2]])
        # reg_smooth_xy = 1e1 # 1e2

        loss_traj = loss_fn(trajec)
        traj_weight = 1e2  # 1e2
        reg = 0.0  # 1e1
        loss_reg = F.mse_loss(x_in, x_in_clone)
        loss_sum = (
            loss_traj * traj_weight + reg * loss_reg +
            # reg_turning * loss_turning +
            # reg_turn_smooth * loss_turn_smooth
            reg_smooth + loss_smooth_traj
            # reg_smooth_xy * loss_smooth_xy
        )

        # losses.append(float(loss_sum))
        if j == 0:
            print('loss traj', float(loss_traj))
            print('loss reg', float(loss_reg))
            # print('loss turn', float(loss_turning))
            print('loss smooth', float(loss_smooth_traj))
            # print('loss turn smooth', float(loss_turn_smooth))
            # print('loss smooth xy', float(loss_smooth_xy))

        opt.zero_grad()
        loss_sum.backward()

        # gradient normalization
        # x_in.grad = F.normalize(x_in.grad, dim=1)

        # cum_grad += x_in.grad.detach()

        # don't need it for now
        if float(loss_sum < 0.01):
            break

        opt.step()

    # import pdb; pdb.set_trace()
    print('after opt loss traj', float(loss_traj))
    print('after opt loss reg', float(loss_reg))
    print('after opt loss smooth', float(loss_smooth_traj))
    # print('after opt loss turn', float(loss_turning))
    # print('after opt loss turn smooth', float(loss_turn_smooth))
    # print('after opt loss smooth xy', float(loss_smooth_xy))
    print("grad size", float(torch.norm(x_in - x_in_clone)))
    print('----')

    # Replace the first three value with the one after opt, keep the rest the same.
    x_last_adjust = inv_transform(x_in.permute(0, 2, 3, 1))  # [1, 1, 120, 263]
    x_in_clone_adjust = inv_transform(x_in_clone.permute(
        0, 2, 3, 1))  # [1, 1, 120, 263]
    x_out = x_in_clone_adjust.detach()
    x_out[:, :, :, :3] = x_last_adjust[:, :, :, :3].detach()
    x_out = transform(x_out).permute(0, 3, 1, 2)

    # x_out = x_in

    # cum_grad = x_in - x_in_clone
    VIS_TRAJ = False
    if VIS_TRAJ:
        with torch.no_grad():
            plt.scatter(trajec_start[:, 0].detach().cpu(),
                        trajec_start[:, 2].detach().cpu())
            plt.title("After - Red")

            # x_new = x_in_clone + cum_grad
            x_new_adjust = inv_transform(x_in.permute(0, 2, 3, 1))
            trajec_new = recover_from_ric(x_new_adjust, 22,
                                          abs_3d=abs_3d)[0, 0, :,
                                                         0, :]  # [120, 3]
            plt.scatter(trajec_new[:, 0].detach().cpu(),
                        trajec_new[:, 2].detach().cpu(),
                        color='r')
            c = 1.1
            plt.xlim(-c, 3 + c)
            plt.ylim(-c, 3 + c)
            plt.show()
    return x_out.detach()