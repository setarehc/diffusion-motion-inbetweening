import numpy as np

# NeMF uses target = [pos, rotmat ,trans, root_orient] but we use everything except velocities and contacts


# Matrix that shows joint correspondces to SMPL features
MAT_POS = np.zeros((24, 764), dtype=np.bool)
MAT_POS[0, :3] = True # root position = trans
for joint_idx in range(24):
    ub = 3 + 24*3*3 + 3 * (joint_idx + 1)
    lb = ub - 3
    MAT_POS[joint_idx, lb:ub] = True # joint position = pos

MAT_ROTMAT = np.zeros((24, 764), dtype=np.bool) # rotmat = 24,3,3 wrp to the parent joint
for joint_idx in range(24):
    ub = 3 + 3*3 * (joint_idx + 1)
    lb = ub - 9
    MAT_ROTMAT[joint_idx, lb:ub] = True # joint rotation = rotmat

MAT_HEIGHT = np.zeros((24, 764), dtype=np.bool) # height = 24
for joint_idx in range(24):
    ub = 3 + 24*3*3 + 24*3 + 24*3 + 8 + (joint_idx + 1)
    lb = ub - 1
    MAT_HEIGHT[joint_idx, lb:ub] = True # joint rotation = rotmat

MAT_ROT6D = np.zeros((24, 764), dtype=np.bool) # rot2d = 24,2 wrp to the parent joint
for joint_idx in range(24):
    ub = 3 + 24*3*3 + 24*3 + 24*3 + 8 + 24 + 3 + 24*3 + 24*6 + 6 + 6 * (joint_idx + 1)
    lb = ub - 6
    MAT_ROT6D[joint_idx, lb:ub] = True # joint rotation = rotmat

MAT_ROT = np.zeros((24, 764), dtype=np.bool) # global_xform = 24, 6 wrp to the root
lb = 3 + 24*3*3 + 24*3 + 24*3 + 8 + 24 + 3 + 24*3 + 24*6
MAT_ROT[0, lb:lb+6] = True # root rotation = root_orient
for joint_idx in range(24):
    ub = 3 + 24*3*3 + 24*3 + 24*3 + 8 + 24 + 3 + 24*3 + (joint_idx + 1) * 6
    lb = ub - 6
    MAT_ROT[joint_idx, lb:ub] = True # joint rotation = global_xform