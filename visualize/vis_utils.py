from model.rotation2xyz import Rotation2xyz
import numpy as np
import trimesh
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'][0].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        ## if there are just 3 features (xyz?) run SMPLify which updates the motion to a 6-feature model
        ## TODO: figure out what the 3 features used here are
        if self.nfeats == 3:
            print(f'Running SMPLify for sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            print("NOTE: This converts rotations to a 6D representation and adds 2 'joints', e.g. 22x3 -> 24x6")
            print("NOTE: This then adds root node locations, e.g. 24x6 + 1x6 -> 25x6")

            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][0][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()        # how does this change to (1, 25, 6, 196) from (1, 22, 3, 196)?? See NOTEs above.
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][0][self.absl_idx]

        print("NOTE: Finally, the 6D motion is converted back to xyz (3 dimensions) here.")
        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def get_traj_sphere(self, mesh):
        root_posi = np.copy(mesh.vertices).mean(0) # (6000, 3)
        root_posi[1]  = self.vertices.numpy().min(axis=(0, 1, 3))[1] + 0.1      # why use min() from 3 axes?
        mesh = trimesh.primitives.Sphere(radius=0.05, center=root_posi, transform=None, subdivisions=1)
        return mesh

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        ground_sph_mesh = self.get_traj_sphere(mesh)
        loc_obj_name = os.path.splitext(os.path.basename(save_path))[0] + "_ground_loc.obj"
        ground_save_path = os.path.join(os.path.dirname(save_path), "loc", loc_obj_name)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        with open(ground_save_path, 'w') as fw:
            ground_sph_mesh.export(fw, 'obj')
        return save_path

    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)
