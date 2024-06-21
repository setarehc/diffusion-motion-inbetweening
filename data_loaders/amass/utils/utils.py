import os
from data_loaders.amass.utils.fk import ForwardKinematicsLayer
from data_loaders.amass.utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix

from data_loaders.humanml.common.quaternion import *
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from data_loaders.amass.utils.helper_functions import estimate_angular_velocity, estimate_linear_velocity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fk = ForwardKinematicsLayer()
fps = 30
root_transform = True
v_axis = [0, 1]


def dict_to_batch(data_dict):
    data = []
    batch_size, n_frames = data_dict['pos'].shape[:2]
    for key, value in data_dict.items():
        assert value.shape[0] == batch_size and value.shape[1] == n_frames
        data.append(value.reshape(batch_size, n_frames, -1))
    data = torch.cat(data, dim=-1)
    return data.unsqueeze(1)


def transform_one(gt_data, data, save_path=None):
    assert gt_data['rotmat'].shape == data['rotmat'].shape
    b_size, n_frames, n_joints = data['rotmat'].shape[:3]

    saved_data = {}

    save_to_path = save_path is not None

    rotmat = data['rotmat']  # (B, T, J, 3, 3)

    b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.reshape(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)

    root_orient = rotation_6d_to_matrix(data['root_orient'])  # (B, T, 3, 3)
    if root_transform: #True
        local_rotmat[:, :, 0] = root_orient

    origin = gt_data['trans'][:, 0]
    trans = data['trans']
    trans[..., v_axis] = trans[..., v_axis] + origin[..., v_axis].unsqueeze(1)

    if save_to_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    all_poses = np.zeros((b_size, n_frames, 165))
    all_trans = np.zeros((b_size, n_frames, 3))
    all_betas = np.zeros((b_size, 10))

    for i in range(b_size):
        poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
        poses = poses.reshape((poses.shape[0], -1))  # (T, 66)
        poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')

        trans = c2c(trans[i])

        all_poses[i] = poses
        all_trans[i] = trans

        if save_to_path:
            offset = 0
            np.savez(os.path.join(save_path, f'recon_{offset + i:03d}_{fps}fps.npz'),
                        poses=poses, trans=c2c(trans[i]), betas=np.zeros(10), gender='male', mocap_framerate=fps)

        saved_data['poses'] = all_poses
        saved_data['trans'] = all_trans
        saved_data['betas'] = all_betas
        saved_data['gender'] = 'male'
        saved_data['mocap_framerate'] = fps

        return saved_data


def save_data(data, save_path=None, gt=True):
    b_size, n_frames, n_joints = data['rotmat'].shape[:3]

    saved_data = {}

    save_to_path = save_path is not None

    rotmat = data['rotmat'] # (B, T, J, 3, 3)
    local_rotmat = fk.global_to_local(rotmat.reshape(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3) # (B, T, J, 3, 3)

    root_orient = rotation_6d_to_matrix(data['root_orient'])  # (B, T, 3, 3)
    if root_transform: #True
        local_rotmat[:, :, 0] = root_orient

    if save_to_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    all_poses = np.zeros((b_size, n_frames, 165))
    all_trans = np.zeros((b_size, n_frames, 3))
    all_betas = np.zeros((b_size, 10))

    for i in range(b_size):
        poses = c2c(matrix_to_axis_angle(local_rotmat[i])) # (T, J, 3)
        poses = poses.reshape((poses.shape[0], -1))  # (T, 72)
        poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant') # (T, 165)
        trans = c2c(data['trans'][i])

        all_poses[i] = poses
        all_trans[i] = trans

        if save_to_path:
            offset = 0
            save_name = f'recon_{offset + i:03d}_gt.npz' if gt else f'recon_{offset + i:03d}_{fps}fps.npz'
            np.savez(os.path.join(save_path, save_name),
                        poses=poses, trans=trans, betas=np.zeros(10), gender='male', mocap_framerate=fps)

    saved_data['poses'] = all_poses
    saved_data['trans'] = all_trans
    saved_data['betas'] = all_betas
    saved_data['gender'] = 'male'
    saved_data['mocap_framerate'] = fps

    return saved_data


def prep_to_save(data):
    # prepares data from dataset format to save format - T1
    saved_data = {}

    b_size, n_frames, n_joints = data['rotmat'].shape[:3]

    rotmat = data['rotmat'] # (B, T, J, 3, 3)
    local_rotmat = fk.global_to_local(rotmat.reshape(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3) # (B, T, J, 3, 3)

    root_orient = rotation_6d_to_matrix(data['root_orient'])  # (B, T, 3, 3)
    if root_transform: #True
        local_rotmat[:, :, 0] = root_orient

    all_poses = np.zeros((b_size, n_frames, 165))
    all_trans = np.zeros((b_size, n_frames, 3))
    all_betas = np.zeros((b_size, 10))

    for i in range(b_size):
        poses = c2c(matrix_to_axis_angle(local_rotmat[i])) # (T, J, 3)
        poses = poses.reshape((poses.shape[0], -1))  # (T, 72)
        poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant') # (T, 165)
        trans = c2c(data['trans'][i])

        all_poses[i] = poses
        all_trans[i] = trans

    saved_data['poses'] = all_poses
    saved_data['trans'] = all_trans
    saved_data['betas'] = all_betas
    saved_data['gender'] = 'male'
    saved_data['mocap_framerate'] = fps

    return saved_data


def load_data(files, max_samples=400):
    poses = []
    trans = []
    assert len(files) != 0, 'files not found'

    max_samples = min(max_samples, len(files))
    for f in files[:max_samples]:
        bdata = np.load(f)
        if 'poses' in bdata.keys():
            poses.append(bdata['poses'][:, :72])
        elif 'root_orient' in bdata.keys() and 'pose_body' in bdata.keys():
            root_orient = bdata['root_orient']
            pose_body = bdata['pose_body']
            poses.append(np.concatenate((root_orient, pose_body), axis=-1))
        else:
            raise RuntimeError(f'missing pose parameters in the file: {f}')
        trans.append(bdata['trans'])

    trans = torch.from_numpy(np.asarray(trans, np.float32)).to(device)  # global translation (N, T, 3)
    N, T = trans.shape[:2]
    poses = torch.from_numpy(np.asarray(poses, np.float32)).to(device)
    poses = poses.view(N, T, 24, 3)  # axis-angle (N, T, J, 3)

    root_orient = poses[:, :, 0].clone()
    root_rotation = axis_angle_to_matrix(root_orient)  # (N, T, 3, 3)
    poses[:, :, 0] = 0

    rotmat = axis_angle_to_matrix(poses)  # (N, T, J, 3, 3)
    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / fps)  # angular velocity of all the joints (N, T, J, 3)
    pos, global_xform = fk(rotmat.view(-1, 24, 3, 3))
    pos = pos.contiguous().view(N, T, 24, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(N, T, 24, 4, 4)
    global_xform = global_xform[:, :, :, :3, :3]  # global transformation matrix for each joint (N, T, J, 3, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / fps)  # linear velocity of all the joints (N, T, J, 3)

    root_vel = estimate_linear_velocity(trans, dt=1.0 / fps)  # linear velocity of the root joint (N, T, 3)

    global_pos = torch.matmul(root_rotation.unsqueeze(2), pos.unsqueeze(-1)).squeeze(-1)  # (N, T, J, 3)
    global_pos = global_pos + trans.unsqueeze(2)

    data = {
        'pos': pos,
        'velocity': velocity,
        'global_xform': matrix_to_rotation_6d(global_xform),
        'angular': angular,
        'root_orient': matrix_to_rotation_6d(root_rotation),
        'root_vel': root_vel,
        'global_pos': global_pos,
        'rotmat': rotmat,
        'trans': trans
    }

    return data



def prep_to_load(data):
    # prepares data from save format to loaded data format - T2
    # this is what inputed to the prior model for FID/FS score computations
    trans = torch.from_numpy(data['trans']).to(device)  # global translation (B, T, 3)
    b_size, n_frames = trans.shape[:2]
    poses = torch.from_numpy(data['poses']).to(device)  # axis-angle (B, T, 165)
    poses = poses[:,:,:24*3].reshape(b_size, n_frames, 24, 3)  # axis-angle (B, T, J, 3)

    root_orient = poses[:, :, 0].clone()
    root_rotation = axis_angle_to_matrix(root_orient)  # (B, T, 3, 3)
    poses[:, :, 0] = 0

    rotmat = axis_angle_to_matrix(poses)  # (B, T, J, 3, 3)
    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / fps)  # angular velocity of all the joints (B, T, J, 3)
    pos, global_xform = fk(rotmat.view(-1, 24, 3, 3))
    pos = pos.contiguous().view(b_size, n_frames, 24, 3)  # local joint positions (B, T, J, 3)
    global_xform = global_xform.view(b_size, n_frames, 24, 4, 4)
    global_xform = global_xform[:, :, :, :3, :3]  # global transformation matrix for each joint (B, T, J, 3, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / fps)  # linear velocity of all the joints (B, T, J, 3)

    root_vel = estimate_linear_velocity(trans, dt=1.0 / fps)  # linear velocity of the root joint (B, T, 3)

    global_pos = torch.matmul(root_rotation.unsqueeze(2), pos.unsqueeze(-1)).squeeze(-1)  # (B, T, J, 3)
    global_pos = global_pos + trans.unsqueeze(2)

    data = {
        'pos': pos,
        'velocity': velocity,
        'global_xform': matrix_to_rotation_6d(global_xform),
        'angular': angular,
        'root_orient': matrix_to_rotation_6d(root_rotation),
        'root_vel': root_vel,
        'global_pos': global_pos,
        'rotmat': rotmat,
        'trans': trans
    }

    return data


def batch_to_dict(batch):
    # batch.shape = (batch_size, 1, 128, 764)
    batch_size, nfeats, nframes, njoints = batch.shape

    assert njoints == 764 # TODO: write for args.amass_fields == 'hml'

    data_dict = {}

    data_dict['trans'] = batch[..., 0:3].squeeze(1)
    data_dict['rotmat'] = batch[..., 3:3+24*3*3].reshape(batch_size, nfeats, nframes, 24, 3, 3).squeeze(1) # redundant
    data_dict['pos'] = batch[..., 219:219+24*3].reshape(batch_size, nfeats, nframes, 24, 3).squeeze(1)
    data_dict['angular'] = batch[..., 291:291+24*3].reshape(batch_size, nfeats, nframes, 24, 3).squeeze(1)
    data_dict['contacts'] = batch[..., 363:363+8].squeeze(1)
    data_dict['height'] = batch[..., 371:371+24*1].squeeze(1)
    data_dict['root_vel'] = batch[..., 395:395+3].squeeze(1)
    data_dict['velocity'] = batch[..., 398:398+24*3].reshape(batch_size, nfeats, nframes, 24, 3).squeeze(1)
    data_dict['global_xform'] = batch[..., 470:470+24*6].reshape(batch_size, nfeats, nframes, 24, 6).squeeze(1)
    data_dict['root_orient'] = batch[..., 614:614+6].squeeze(1)
    data_dict['rot6d'] = batch[..., 620:].reshape(batch_size, nfeats, nframes, 24, 6).squeeze(1)

    return data_dict


def dict_to_xyz(data_dict):
    r_rot_quat = cont6d_to_quaternion(data_dict['root_orient']) # (B, T, 4)
    r_pos = data_dict['trans'] # (B, T, 3)
    # TODO: figure out height

    positions = data_dict['pos'] # (B, T, 24, 3)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    # TODO: might need to do y instead of z
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 1] += r_pos[..., 1:2]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    # positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    positions[..., :1,:] = r_pos.unsqueeze(-2)
    positions[..., 1] = data_dict['height']

    return positions


def cont6d_to_quaternion(cont6d):
    rot_mat = rotation_6d_to_matrix(cont6d)
    rot_quat = matrix_to_quaternion(rot_mat)
    return rot_quat


def dict_to_posrot(data_dict):
    fk = ForwardKinematicsLayer()
    b_size, nframes, n_joints, dim = data_dict['pos'].shape # (B, T, J, 3)
    local_rotmat = fk.global_to_local(data_dict['rotmat'].reshape(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.reshape(b_size, -1, n_joints, 3, 3)

    root_orient = rotation_6d_to_matrix(data_dict['root_orient'])  # (B, T, 3, 3)
    local_rotmat[:, :, 0] = root_orient
    rotations = matrix_to_quaternion(local_rotmat)

    pos = data_dict['pos']

    # origin = self.input_data['trans'][:, 0]
    positions = data_dict['trans']
    # positions[..., self.v_axis] = positions[..., self.v_axis] + origin[..., self.v_axis].unsqueeze(1)

    return positions, rotations
