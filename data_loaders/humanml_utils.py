import numpy as np

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]
HML_LOWER_BODY_RIGHT_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'right_foot',]]
HML_PELVIS_FEET = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_foot', 'right_foot']]
HML_PELVIS_HANDS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_wrist', 'right_wrist']]
HML_PELVIS_VR = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_wrist', 'right_wrist', 'head']]

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

HML_LOWER_BODY_RIGHT_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_RIGHT_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_RIGHT_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_RIGHT_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_RIGHT_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_RIGHT_JOINTS_BINARY.repeat(3),
                                     [True]*4))


# Matrix that shows joint correspondces to SMPL features
MAT_POS = np.zeros((22, 263), dtype=np.bool)
MAT_POS[0, 1:4] = True
for joint_idx in range(1, 22):
    ub = 4 + 3 * joint_idx
    lb = ub - 3
    MAT_POS[joint_idx, lb:ub] = True

MAT_ROT = np.zeros((22, 263), dtype=np.bool)
MAT_ROT[0, 0] = True
for joint_idx in range(1, 22):
    ub = 4 + 21*3 + 6 * joint_idx
    lb = ub - 6
    MAT_ROT[joint_idx, lb:ub] = True

MAT_VEL = np.zeros((22, 263), dtype=np.bool)
for joint_idx in range(0, 22):
    ub = 4 + 21*3 + 21*6 + 3 * (joint_idx + 1)
    lb = ub - 3
    MAT_VEL[joint_idx, lb:ub] = True

MAT_CNT = np.zeros((22, 263), dtype=np.bool)
MAT_CNT[7, -4] = True
MAT_CNT[10, -3] = True
MAT_CNT[8, -2] = True
MAT_CNT[11, -1] = True