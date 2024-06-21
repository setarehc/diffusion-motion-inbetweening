### Helper functions adopted from src/utils.py in NeMF

import torch

def estimate_linear_velocity(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def estimate_angular_velocity(rot_seq, dt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w