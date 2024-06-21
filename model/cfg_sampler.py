import torch.nn as nn
from copy import deepcopy


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.keyframe_conditioned = self.model.keyframe_conditioned

        # self.mask_value = 0.0
        self.mask_value = -2.0

    def forward(self, x, timesteps, y=None, obs_x0=None, obs_mask=None, **kwargs):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # If there is condition, the uncond_model will take in the spatial condition as well
        out = self.model(x, timesteps, y, obs_x0, obs_mask, **kwargs)
        out_uncond = self.model(x, timesteps, y_uncond, obs_x0, obs_mask, **kwargs)
        # return out
        # return out_uncond + (1.5 * (out - out_uncond))
        return out_uncond + (y['text_scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def forward_smd_final(self, x, timesteps, y=None, **kwargs):
        '''_ori'''
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # If there is condition, the uncond_model will take in the spatial condition as well
        out = self.model(x, timesteps, y, **kwargs)
        out_uncond = self.model(x, timesteps, y_uncond, **kwargs)
        # return out
        return out_uncond + (1.5 * (out - out_uncond))
        # return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def forward_correct(self, x, timesteps, y=None):
        '''_correct'''
        kps_to_text_ratio = 0.8
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # out = self.model(x, timesteps, y)
        # out_spatial = self.model(x, timesteps, y_uncond)

        x_without_spatial = x + 0.0
        x_without_spatial[:, 263, :, :] *= 0.0
        x_without_spatial[:, 264:, :, :] = self.mask_value
        # out_uncond = self.model(x_without_spatial, timesteps, y_uncond)
        out_uncond = self.model(x, timesteps, y_uncond)
        # return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
        # out = out_uncond + (0.9 * (out_spatial - out_uncond))
        # out_text = self.model(x_without_spatial, timesteps, y)
        out_text = self.model(x, timesteps, y)

        out = out_uncond + (1.8 * (out_text - out_uncond))
        # out = out_uncond + (0.5 * (out_text_scale - out_uncond))
        # out[:, :3, :, :] = out_text[:, :3, :, :]
        return out # out_uncond + (0.8 * (out_patial - out_uncond))
        # return out_uncond + (2.5 * (out - out_uncond))


    def forward_average(self, x, timesteps, y=None):
        '''_average'''
        kps_to_text_ratio =  1.5
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out_spatial = self.model(x, timesteps, y_uncond)

        x_without_spatial = x + 0.0
        x_without_spatial[:, 263, :, :] *= 0.0
        x_without_spatial[:, 264:, :, :] = self.mask_value
        out_uncond = self.model(x_without_spatial, timesteps, y_uncond)
        # out_uncond = self.model(x, timesteps, y_uncond)
        # out_with_spatial = out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

        out_text = self.model(x_without_spatial, timesteps, y)
        # out_text = self.model(x, timesteps, y)

        combined_out_spatial = out_uncond + (1.0 * (out_spatial - out_uncond))
        combined_out_text = out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out_text - out_uncond))

        # return 1.0 * combined_out_text + 1.0 * combined_out_spatial - 1.0 * out_uncond
        return combined_out_text + (kps_to_text_ratio) * (combined_out_spatial - combined_out_text)


        # x_without_spatial = x + 0.0
        # x_without_spatial[:, 263:, :, :] *= 0.0
        # out_no_cond = self.model(x_without_spatial, timesteps, y)
        # out_uncond_without_spatial = self.model(x_without_spatial, timesteps, y_uncond)
        # out_without_spatial = out_uncond_without_spatial + (y['scale'].view(-1, 1, 1, 1) * (out_no_cond - out_uncond_without_spatial))

        # # import pdb; pdb.set_trace()
        # return out_without_spatial + (kps_to_text_ratio) * (out_with_spatial - out_without_spatial)
