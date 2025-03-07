import numpy as np

from data_loaders.custom.scripts.motion_process import fid_l, fid_r

RIG_JOINT_NAMES = [
    "Root",
    "Spine",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "RightToe_end",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "LeftToe_end",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "Head_end",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHand_end",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHand_end"
]

NUM_RIG_JOINTS = len(RIG_JOINT_NAMES)  # joints in the custom rig
NUM_RIG_FEATURES = 12 * NUM_RIG_JOINTS - 1    # precalculate the features needed for this rig

RIG_LOWER_BODY_JOINTS = [RIG_JOINT_NAMES.index(name) for name in ['Root', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe', 'RightToe_end', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 'LeftToe_end']]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(RIG_JOINT_NAMES)) if i not in RIG_LOWER_BODY_JOINTS]
RIG_LOWER_BODY_RIGHT_JOINTS = [RIG_JOINT_NAMES.index(name) for name in ['Root', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe', 'RightToe_end']]
RIG_PELVIS_FEET = [RIG_JOINT_NAMES.index(name) for name in ['Root', 'LeftFoot', 'RightFoot']]
RIG_PELVIS_HANDS = [RIG_JOINT_NAMES.index(name) for name in ['Root', 'LeftHand', 'RightHand']]
RIG_PELVIS_VR = [RIG_JOINT_NAMES.index(name) for name in ['Root', 'LeftHand', 'RightHand', 'Head']]

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
RIG_ROOT_BINARY = np.array([True] + [False] * (NUM_RIG_JOINTS-1))
RIG_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                RIG_ROOT_BINARY[1:].repeat(3),
                                RIG_ROOT_BINARY[1:].repeat(6),
                                RIG_ROOT_BINARY.repeat(3),
                                [False] * 4))
RIG_LOWER_BODY_JOINTS_BINARY = np.array([i in RIG_LOWER_BODY_JOINTS for i in range(NUM_RIG_JOINTS)])
RIG_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     RIG_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     RIG_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     RIG_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
RIG_UPPER_BODY_MASK = ~RIG_LOWER_BODY_MASK

RIG_LOWER_BODY_RIGHT_JOINTS_BINARY = np.array([i in RIG_LOWER_BODY_RIGHT_JOINTS for i in range(NUM_RIG_JOINTS)])
RIG_LOWER_BODY_RIGHT_MASK = np.concatenate(([True]*(1+2+1),
                                     RIG_LOWER_BODY_RIGHT_JOINTS_BINARY[1:].repeat(3),
                                     RIG_LOWER_BODY_RIGHT_JOINTS_BINARY[1:].repeat(6),
                                     RIG_LOWER_BODY_RIGHT_JOINTS_BINARY.repeat(3),
                                     [True]*4))


# Matrix that shows joint correspondces to SMPL features
MAT_POS = np.zeros((NUM_RIG_JOINTS, NUM_RIG_FEATURES), dtype=bool)
MAT_POS[0, 1:4] = True
for joint_idx in range(1, NUM_RIG_JOINTS):
    ub = 4 + 3 * joint_idx
    lb = ub - 3
    MAT_POS[joint_idx, lb:ub] = True

MAT_ROT = np.zeros((NUM_RIG_JOINTS, NUM_RIG_FEATURES), dtype=bool)
MAT_ROT[0, 0] = True
for joint_idx in range(1, NUM_RIG_JOINTS):
    ub = 4 + (NUM_RIG_JOINTS - 1)*3 + 6 * joint_idx
    lb = ub - 6
    MAT_ROT[joint_idx, lb:ub] = True

MAT_VEL = np.zeros((NUM_RIG_JOINTS, NUM_RIG_FEATURES), dtype=bool)
for joint_idx in range(0, NUM_RIG_JOINTS):
    ub = 4 + (NUM_RIG_JOINTS - 1)*3 + (NUM_RIG_JOINTS -1)*6 + 3 * (joint_idx + 1)
    lb = ub - 3
    MAT_VEL[joint_idx, lb:ub] = True

MAT_CNT = np.zeros((NUM_RIG_JOINTS, NUM_RIG_FEATURES), dtype=bool)

## Feet contacts are different for each rig, so we import from scripts/motion_process
MAT_CNT[fid_l[0], -4] = True
MAT_CNT[fid_l[1], -3] = True
MAT_CNT[fid_r[0], -2] = True
MAT_CNT[fid_r[1], -1] = True
