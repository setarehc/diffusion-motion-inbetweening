# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from dataclasses import dataclass
import enum
import math
import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from data_loaders.humanml.scripts import motion_process
from torch.cuda import amp
from typing import List
import utils.editing_util as inpainting_util


def get_named_beta_schedule(schedule_name,
                            num_diffusion_timesteps,
                            scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


@dataclass
class DiffusionConfig:
    betas: List
    model_mean_type: ModelMeanType = ModelMeanType.START_X
    model_var_type: ModelVarType = ModelVarType.FIXED_SMALL
    loss_type: LossType = LossType.MSE
    rescale_timesteps: bool = False
    lambda_rcxyz: float = 0.
    lambda_vel: float = 0.
    lambda_pose: float = 1.
    lambda_orient: float = 1.
    lambda_loc: float = 1.
    data_rep: str = 'rot6d'
    lambda_root_vel: float = 0.
    lambda_vel_rcxyz: float = 0.
    lambda_fc: float = 0.
    clip_range: float = None
    train_trajectory_only_xz: bool = False
    use_random_proj: bool = False
    fp16: bool = False
    traj_only: bool = False
    abs_3d: bool = False
    apply_zero_mask: bool = False
    traj_extra_weight: float = 1.
    time_weighted_loss: bool = False
    train_x0_as_eps: bool = False
    train_keypoint_mask: str = 'none'


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, conf: DiffusionConfig):
        self.conf = conf

        # TODO: to removed and use only self.conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps
        self.data_rep = conf.data_rep
        self.clip_range = conf.clip_range

        if conf.data_rep != 'rot_vel' and conf.lambda_pose != 1.:
            raise ValueError(
                'lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = conf.lambda_pose
        self.lambda_orient = conf.lambda_orient
        self.lambda_loc = conf.lambda_loc

        self.lambda_rcxyz = conf.lambda_rcxyz
        self.lambda_vel = conf.lambda_vel
        self.lambda_root_vel = conf.lambda_root_vel
        self.lambda_vel_rcxyz = conf.lambda_vel_rcxyz
        self.lambda_fc = conf.lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))

        # for time-weighted loss
        c = np.zeros_like(betas)
        c[1:] = (1 - self.alphas_cumprod[:-1]) / (
            1 - self.alphas_cumprod[1:]) * np.sqrt(alphas[1:])
        d = np.zeros_like(betas)

        d[1:] = self.sqrt_alphas_cumprod[:-1] / (
            1 - self.alphas_cumprod[1:]) * betas[1:]
        e = c + d
        f = d * self.sqrt_one_minus_alphas_cumprod / np.sqrt(
            self.alphas_cumprod)
        self.ratio_eps = f / (e + f + 1e-8)

        self.sqrt_alphas_cumprod_over_oneminus_aphas_cumprod = self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod

        self.l2_loss = lambda a, b: (
            a - b
        )**2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

        self.data_transform_fn = None
        self.data_inv_transform_fn = None
        self.data_get_mean_fn = None
        self.log_trajectory_fn = None

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(
            loss *
            mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    def masked_l2_weighted(self, a, b, mask, weights, time_weights, over_keyframes=False):
        """
        Args:
            a: bs, J, Jdim, seqlen
            b: bs, J, Jdim, seqlen
            mask: bs, 1, 1, seqlen
            weights: bs, J, Jdim, 1
            time_weights: bs, J, Jdim, seqlen
        """
        assert ((mask.shape == (a.shape[0], 1, 1, a.shape[3]) and not over_keyframes) or (mask.shape == a.shape and over_keyframes))
        assert weights.shape == (a.shape[0], a.shape[1], a.shape[2],
                                 1), f'{weights.shape}'
        assert time_weights.shape == (a.shape[0], a.shape[1], a.shape[2],
                                 a.shape[3]), f'{time_weights.shape}'

        loss = self.l2_loss(a, b)
        # average over the features, dim [1,2]
        weights = weights / weights.sum(dim=[1, 2], keepdims=True)
        loss = loss * weights

        # time_weights = time_weights / time_weights.sum(dim=[3], keepdims=True)
        loss = loss * time_weights

        # [bs, J, Jdim, seqlen]
        loss = loss * mask.float(
        )  # gives \sigma_euclidean over unmasked elements
        # n_entries = a.shape[1] * a.shape[2]
        # [bs]
        loss = sum_flat(loss)
        # average over the length
        # [bs]
        non_zero_elements = sum_flat(mask)
        loss = loss / non_zero_elements
        return loss

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                        x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                            t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            previous_xstart=None, # NOTE: for debugging
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B, )

        def get_conv_model_output(x, t, model_kwargs):
            with amp.autocast(enabled=self.conf.fp16):

                if self.conf.train_keypoint_mask != 'none':
                    pass
                    # return

                else:
                    model_output = model(x, self._scale_timesteps(t), **model_kwargs) # DEFALUT
            if isinstance(model_output, tuple): # By default, is False
                # NOTE: in this case the model has two heads
                # the first head predicts eps
                # the second head predicts x0
                # x0 is ignored here but used for guidance later
                model_output, _ = model_output
            return model_output

        if inpainting_util.requires_reconstruction_guidance(model_kwargs, t):
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for now!'
            mask = model_kwargs['y']['mask'].float().to(inpainting_mask.device)
            inpainting_mask = (inpainting_mask * mask).bool()
            # Perform reconstruction guidance
            with torch.enable_grad():
                z = x.detach().requires_grad_(True)
                hat_x = get_conv_model_output(z, t, model_kwargs)
                assert hat_x.shape == inpainting_mask.shape == inpainted_motion.shape
                guidance_loss = ((inpainted_motion - hat_x).square() * inpainting_mask).sum()
                cond_grad = torch.autograd.grad(guidance_loss, z)[0] * (~inpainting_mask).float()

            grad_ws = inpainting_util.get_gradient_schedule(schedule_name=model_kwargs['y']['gradient_schedule'],
                                                                    num_diffusion_steps=model_kwargs['y']['diffusion_steps'])
            w_r = _extract_into_tensor(grad_ws, t, cond_grad.shape) * model_kwargs['y']['reconstruction_weight']
            sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, cond_grad.shape)
            tilde_x = hat_x - (w_r * sqrt_alpha_bar  / 2) * cond_grad

            model_output = (tilde_x * ~inpainting_mask) + (inpainted_motion * inpainting_mask) if inpainting_util.requires_imputation(model_kwargs, t) else \
                           (tilde_x * ~inpainting_mask) + (hat_x * inpainting_mask)

        elif inpainting_util.requires_imputation(model_kwargs, t):
            if model_kwargs['y']['replacement_distribution'] == 'conditional':
                # Inpainting where noise added to observed data comes from q(x_{t-1} | x_0, x_t)
                inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
                assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for now!'
                mask = model_kwargs['y']['mask'].float().to(inpainting_mask.device)
                inpainting_mask = (inpainting_mask * mask).bool()
                hat_x = get_conv_model_output(x, t, model_kwargs)
                model_output = (hat_x * ~inpainting_mask) + (inpainted_motion * inpainting_mask)

            elif model_kwargs['y']['replacement_distribution'] == 'marginal':
                # Inpainting where noise added to observed data comes from q(x_{t-1} | x_0) - implementation in p_sample_loop_progressive
                model_output = get_conv_model_output(x, t, model_kwargs)

            else:
                raise NotImplementedError

        else:
            model_output = get_conv_model_output(x, t, model_kwargs)

        denoised_fn = None

        if self.model_var_type in [
                ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE
        ]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t,
                                                      x.shape)

        def process_xstart(x, model_mean_type):
            if denoised_fn is not None:
                x = denoised_fn(x)
            # only applies to epsilon prediction
            # NOTE: it's now safe to always clip!
            if clip_denoised:
                if model_mean_type == ModelMeanType.START_X:
                    # there is no need to clip if the model predicts xstart
                    return x
                else:
                    # TODO: whether to clip should be determined from the outside
                    if self.conf.abs_3d:
                        if self.conf.traj_only:
                            return x.clamp(-self.clip_range, self.clip_range)
                        else:
                            raise NotImplementedError()
                    else:
                        raise NotImplementedError()
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t,
                                                xprev=model_output),
                self.model_mean_type)
            model_mean = model_output
        elif self.model_mean_type in [
                ModelMeanType.START_X, ModelMeanType.EPSILON
        ]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output,
                                             self.model_mean_type)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t,
                                                  eps=model_output),
                    self.model_mean_type)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_output": model_output,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        updated_xstart = cond_fn(p_mean_var["pred_xstart"],
                                    self._scale_timesteps(t),
                                    diffusion=self,
                                    **model_kwargs)
        new_mean, _, _ = self.q_posterior_mean_variance(
            x_start=updated_xstart, x_t=x, t=t)

        return new_mean

    def condition_mean_with_grad(self,
                                 cond_fn,
                                 p_mean_var,
                                 x,
                                 t,
                                 model_kwargs=None,
                                 var_scale=True):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        if var_scale:
            new_mean = (p_mean_var["mean"].float() +
                        p_mean_var["variance"] * gradient.float())
        else:
            new_mean = p_mean_var["mean"].float() + gradient.float(
            )

        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        # NOTE: We modified this such that our cond_fn returns the desired pred_xstart directly

        out = p_mean_var.copy()
        if int(t) <= 1000 and int(t) > 0:
            new_pred_xstart = cond_fn(p_mean_var["pred_xstart"],
                                      self._scale_timesteps(t), **model_kwargs)
            w = 1.
            out["pred_xstart"] = w * new_pred_xstart + (1 -
                                                        w) * out["pred_xstart"]
        else:
            out["pred_xstart"] = p_mean_var["pred_xstart"]
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def condition_score_with_grad(self,
                                  cond_fn,
                                  p_mean_var,
                                  x,
                                  t,
                                  model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        eps = eps - (1 - alpha_bar).sqrt() * gradient
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        previous_xstart=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        assert cond_fn is None, "only support the case where cond_fn is None"

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            previous_xstart=previous_xstart,
        )
        noise = th.randn_like(x)
        # print('const_noise', const_noise)
        if const_noise:
            raise NotImplementedError()
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn,
                                              out,
                                              x,
                                              t,
                                              model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(
            0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        previous_xstart=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # import pdb; pdb.set_trace()
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                previous_xstart=previous_xstart,
            )
            noise = th.randn_like(x)
            nonzero_mask = ((t != 0).float().view(-1,
                                                  *([1] * (len(x.shape) - 1)))
                            )  # no noise when t == 0

            # NOTE: support cond_until and conditioning on the whole trajectory
            if 'inpainting_mask' in model_kwargs['y'].keys(
            ) and 'inpainted_motion' in model_kwargs['y'].keys():
                model_kwargs['y']['current_inpainting_mask'] = model_kwargs[
                    'y']['inpainting_mask']
                model_kwargs['y']['current_inpainted_motion'] = model_kwargs[
                    'y']['inpainted_motion']
            if 'cond_until' in model_kwargs['y'].keys():
                cond_until = model_kwargs['y']['cond_until']
                # Switch to second stage
                if int(
                        t[0]
                ) < cond_until and 'cond_until_second_stage' in model_kwargs[
                        'y'].keys():
                    cond_until = model_kwargs['y']['cond_until_second_stage']
                    model_kwargs['y'][
                        'current_inpainting_mask'] = model_kwargs['y'][
                            'inpainting_mask_second_stage']
                    model_kwargs['y'][
                        'current_inpainted_motion'] = model_kwargs['y'][
                            'inpainted_motion_second_stage']
            else:
                cond_until = 1

            if cond_fn is not None and int(t[0]) >= cond_until:
                # you need to supply with "unaltered x0" and calculate the gradient from it.
                # but you must only update mean in the "inpainting region" not outside (which will interfere with the imposing region).
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn,
                    out,
                    x,
                    t,
                    model_kwargs=
                    model_kwargs,
                )
        sample = out["mean"] + nonzero_mask * th.exp(
            0.5 * out["log_variance"]) * noise

        # We do impainting here
        if 'inpainting_mask' in model_kwargs['y'].keys(
        ) and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y'][
                'inpainting_mask'], model_kwargs['y']['inpainted_motion']
            # We have the option of stop imputing near the end of the denoising process (t -> 0)
            if 'impute_until' in model_kwargs['y'].keys():
                impute_until = model_kwargs['y']['impute_until']
                # Switch to second stage
                if int(
                        t[0]
                ) < impute_until and 'impute_until_second_stage' in model_kwargs[
                        'y'].keys():
                    impute_until = model_kwargs['y'][
                        'impute_until_second_stage']
                    inpainting_mask = model_kwargs['y'][
                        'inpainting_mask_second_stage']
                    inpainted_motion = model_kwargs['y'][
                        'inpainted_motion_second_stage']
            else:
                impute_until = 1
            assert sample.shape == inpainting_mask.shape == inpainted_motion.shape

            def impute(a, b, impute_mask):
                '''Override a with b. Mask is True where we want to override.'''
                return (a * ~impute_mask) + (b * impute_mask)

            if int(
                    t[0]
            ) >= impute_until and self.model_mean_type == ModelMeanType.EPSILON:
                '''If the model predict epsilon, add noise at time t to our target image and impute.
                '''
                impute_at = 'mu'  # mu is the best

                if impute_at == 'mu':
                    # NOTE: this is the best method!

                    impute_type = 'q_sample'  # q_sample is the best
                    if impute_type == 'q_sample':
                        # assert self.model_mean_type == ModelMeanType.EPSILON, 'This feature supports only EPSILON pred for now.'
                        t_minus_one = torch.where(t >= 1, t - 1, t)
                        # Add noise to match t-1 level of noise
                        noised_motion = self.q_sample(inpainted_motion,
                                                      t_minus_one)
                        sample = impute(sample, noised_motion, inpainting_mask)
                        # Also update pred_xstart by replacing the imputed parts, this is not the same as what sample currently is.
                        out["pred_xstart"] = (
                            out["pred_xstart"] * ~inpainting_mask) + (
                                inpainted_motion * inpainting_mask)
                    elif impute_type == 'ddim':
                        eps = self._predict_eps_from_xstart(
                            x, t, inpainted_motion)
                        alpha_bar = _extract_into_tensor(
                            self.alphas_cumprod, t, x.shape)
                        alpha_bar_prev = _extract_into_tensor(
                            self.alphas_cumprod_prev, t, x.shape)
                        sigma = (1 * th.sqrt(
                            (1 - alpha_bar_prev) / (1 - alpha_bar)) *
                                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
                        # Equation 12.
                        impute_signal = (
                            inpainted_motion * th.sqrt(alpha_bar_prev) +
                            th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
                        impute_noise = sigma * noise
                        impute_sample = impute_signal + nonzero_mask * impute_noise
                        sample = impute(sample, impute_sample, inpainting_mask)
                    else:
                        raise NotImplementedError()
                elif impute_at == 'x0':

                    combine_type = 'impute'

                    if combine_type == 'combine':
                        cur_xstart = out["pred_xstart"]
                        painted_model_output = impute(cur_xstart,
                                                      inpainted_motion,
                                                      inpainting_mask)
                        model_mean_imputed, _, _ = self.q_posterior_mean_variance(
                            x_start=painted_model_output, x_t=x, t=t)
                        mu = out["mean"]
                        combined_mean = impute(mu, model_mean_imputed,
                                               inpainting_mask)
                        sample = combined_mean + nonzero_mask * th.exp(
                            0.5 * out["log_variance"]) * noise
                    elif combine_type == 'impute':
                        cur_xstart = out["pred_xstart"]
                        painted_model_output = impute(cur_xstart,
                                                      inpainted_motion,
                                                      inpainting_mask)
                        model_mean_imputed, _, _ = self.q_posterior_mean_variance(
                            x_start=painted_model_output, x_t=x, t=t)
                        sample = model_mean_imputed + nonzero_mask * th.exp(
                            0.5 * out["log_variance"]) * noise
                    else:
                        raise NotImplementedError()
                elif impute_at == 'none':
                    pass
                else:
                    raise NotImplementedError()

            elif int(
                    t[0]
            ) >= impute_until and self.model_mean_type == ModelMeanType.START_X:
                ''' If we do random projection the model predicts x_start, we can inpaint directly onto x_start
                '''
                # assert flag.USE_RANDOM_PROJECTION == False, "Random projection does not supports inpainting after sampling"
                t_minus_one = torch.where(t >= 1, t - 1, t)
                # Add noise to match t-1 level of noise

                if self.conf.use_random_proj:
                    impute_at = 'x0'  # x0 is the best
                    assert self.data_transform_fn is not None

                    if impute_at == 'x0':
                        # NOTE: this is the best method!
                        cur_xstart = out["pred_xstart"]
                        # Project back to motion representation
                        unprojected_x_start = self.data_inv_transform_fn(
                            cur_xstart.permute(0, 2, 3,
                                               1))  # [1, 1, 120, 263])
                        # Inpaint
                        painted_model_output = impute(
                            unprojected_x_start,
                            inpainted_motion.permute(0, 2, 3, 1),
                            inpainting_mask.permute(0, 2, 3, 1))

                        # Do random projection again
                        imputed_xstart = self.data_transform_fn(
                            painted_model_output).permute(0, 3, 1, 2)

                        # Need to compute x_{t-1} from x_start again
                        model_mean_imputed, _, _ = self.q_posterior_mean_variance(
                            x_start=imputed_xstart, x_t=x, t=t)
                        # Combine inpainted sample with grad sample. Ratio is arbitrary
                        # Combine means should be better than combine samples with noises

                        combine_type = 'combine'  # impute is the best

                        if combine_type == 'interpolate':
                            impute_ratio = 0.9
                            combined_mean = (1 - impute_ratio) * out[
                                "mean"] + impute_ratio * model_mean_imputed
                        elif combine_type == 'combine':
                            # NOTE: this seems to be the best!
                            mu = out["mean"]
                            unproj_mu = self.data_inv_transform_fn(
                                mu.permute(0, 2, 3, 1))
                            unproj_model_mean_imputed = self.data_inv_transform_fn(
                                model_mean_imputed.permute(0, 2, 3, 1))
                            unproj_combined_mean = impute(
                                unproj_mu, unproj_model_mean_imputed,
                                inpainting_mask.permute(0, 2, 3, 1))
                            combined_mean = self.data_transform_fn(
                                unproj_combined_mean).permute(0, 3, 1, 2)
                        elif combine_type == 'impute':
                            combined_mean = model_mean_imputed
                        else:
                            raise NotImplementedError()
                        # Combine inpainted sample with grad sample. Ratio is arbitrary
                        sample = combined_mean + nonzero_mask * th.exp(
                            0.5 * out["log_variance"]) * noise

                    elif impute_at == 'mu':
                        # NOTE: no good, imputing on x0 is better
                        noise = th.randn_like(sample)
                        # [1, 263, 1, 120]
                        mu = out["mean"]
                        mu_noise = th.exp(0.5 * out["log_variance"]) * noise
                        # [1, 1, 120, 263]
                        unproj_mu = self.data_inv_transform_fn(
                            mu.permute(0, 2, 3, 1))

                        # [1, 263, 1, 120]
                        signal_type = 'ddim'
                        if signal_type == 'scaled':
                            impute_signal = _extract_into_tensor(
                                self.sqrt_alphas_cumprod, t,
                                inpainted_motion.shape) * inpainted_motion
                        elif signal_type == 'ddim':
                            imputed_xstart = inpainted_motion
                            eps = self._predict_eps_from_xstart(
                                x, t, imputed_xstart)
                            alpha_bar = _extract_into_tensor(
                                self.alphas_cumprod, t, x.shape)
                            alpha_bar_prev = _extract_into_tensor(
                                self.alphas_cumprod_prev, t, x.shape)
                            sigma = (1 * th.sqrt(
                                (1 - alpha_bar_prev) / (1 - alpha_bar)) *
                                     th.sqrt(1 - alpha_bar / alpha_bar_prev))
                            # Equation 12.
                            impute_signal = (
                                imputed_xstart * th.sqrt(alpha_bar_prev) +
                                th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
                        else:
                            raise NotImplementedError()

                        # NOTE: 1 - alpha_cumprod seem to be too large a variance
                        var_type = 'ddim'
                        if var_type == 'var_large':
                            impute_noise = _extract_into_tensor(
                                self.sqrt_one_minus_alphas_cumprod, t,
                                noise.shape) * noise
                        elif var_type == 'var_small':
                            impute_noise = mu_noise
                        elif var_type == 'ddim':
                            impute_noise = sigma * noise
                        else:
                            raise NotImplementedError()

                        # [1, 1, 120, 263]
                        combined_unproj_signal = impute(
                            unproj_mu, impute_signal.permute(0, 2, 3, 1),
                            inpainting_mask.permute(0, 2, 3, 1))
                        # [1, 263, 1, 120]
                        combined_signal = self.data_transform_fn(
                            combined_unproj_signal).permute(0, 3, 1, 2)
                        # [1, 263, 1, 120]
                        combined_noise = impute(mu_noise, impute_noise,
                                                inpainting_mask)
                        # [1, 263, 1, 120]
                        sample = combined_signal + nonzero_mask * combined_noise
                    else:
                        raise NotImplementedError()
                else:
                    IMPUTE_AT_X0 = True
                    if 'impute_relative' in model_kwargs['y'] and model_kwargs['y']['impute_relative']:
                        # For relative model (including MDM), we need to impute at x_t and set x_0 to the target,
                        # because we use pred_x_start as output.
                        IMPUTE_AT_X0 = False

                    # NOTE: All imputing methods assume the given motion is in the motion space (not scaled)
                    if IMPUTE_AT_X0:
                        impute_type = 'combine'
                        if impute_type == 'interpolate':
                            cur_xstart = out["pred_xstart"]
                            # Project back to motion representation
                            unprojected_x_start = self.data_inv_transform_fn(
                                cur_xstart.permute(0, 2, 3,
                                                   1))  # [1, 1, 120, 263])
                            # Inpaint
                            painted_model_output = (
                                (unprojected_x_start *
                                 ~inpainting_mask.permute(0, 2, 3, 1)) +
                                (inpainted_motion.permute(0, 2, 3, 1) *
                                 inpainting_mask.permute(0, 2, 3, 1)))
                            # Do random projection again
                            out["pred_xstart"] = self.data_transform_fn(
                                painted_model_output).permute(0, 3, 1, 2)
                            # Need to compute x_{t-1} from x_start again
                            model_mean_imputed, _, _ = self.q_posterior_mean_variance(
                                x_start=out["pred_xstart"], x_t=x, t=t)
                            # Combine inpainted sample with grad sample. Ratio is arbitrary
                            # Combine means should be better than combine samples with noises
                            impute_ratio = 0.9
                            combined_mean = (1 - impute_ratio) * out[
                                "mean"] + impute_ratio * model_mean_imputed
                            sample = combined_mean + nonzero_mask * th.exp(
                                0.5 * out["log_variance"]) * noise

                        elif impute_type == 'combine':
                            cur_xstart = out["pred_xstart"]
                            # Project back to motion representation
                            unprojected_x_start = self.data_inv_transform_fn(
                                cur_xstart.permute(0, 2, 3,
                                                   1))  # [1, 1, 120, 263])
                            # Inpaint
                            painted_model_output = impute(
                                unprojected_x_start,
                                inpainted_motion.permute(0, 2, 3, 1),
                                inpainting_mask.permute(0, 2, 3, 1))

                            # Do random projection again
                            imputed_xstart = self.data_transform_fn(
                                painted_model_output).permute(0, 3, 1, 2)

                            # Need to compute x_{t-1} from x_start again
                            model_mean_imputed, _, _ = self.q_posterior_mean_variance(
                                x_start=imputed_xstart, x_t=x, t=t)

                            mu = out["mean"]
                            unproj_mu = self.data_inv_transform_fn(
                                mu.permute(0, 2, 3, 1))
                            unproj_model_mean_imputed = self.data_inv_transform_fn(
                                model_mean_imputed.permute(0, 2, 3, 1))
                            unproj_combined_mean = impute(
                                unproj_mu, unproj_model_mean_imputed,
                                inpainting_mask.permute(0, 2, 3, 1))
                            combined_mean = self.data_transform_fn(
                                unproj_combined_mean).permute(0, 3, 1, 2)
                            sample = combined_mean + nonzero_mask * th.exp(
                                0.5 * out["log_variance"]) * noise
                            if int(t[0]) == 0:
                                out["pred_xstart"] = sample
                        else:
                            raise NotImplementedError()
                    else:
                        # Old method for MDM. Impute at x_t instead of x_0
                        # Project the given motion to the model space
                        projected_inpainted_motion = self.data_transform_fn(
                            inpainted_motion.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        # NOTE: This method assume that there is no transformations. Check MDM
                        noised_motion = self.q_sample(projected_inpainted_motion,
                                                      t_minus_one)
                        sample = (sample * ~inpainting_mask) + (
                            noised_motion * inpainting_mask)
                        # Also update pred_xstart by replacing the imputed parts, this is not the same as what sample currently is.
                        out["pred_xstart"] = (
                            out["pred_xstart"] * ~inpainting_mask) + (
                                projected_inpainted_motion * inpainting_mask)

        if self.log_trajectory_fn is not None:
            # NOTE: Here we visualize the predicted trajectory *BEFORE* inpainting and gradient
            out_list = [
                990, 900, 800, 700, 600, 500, 400, 300, 200, 100, 10, 0
            ]
            if int(t[0]) in out_list:
                self.log_trajectory_fn(out["pred_xstart"].detach(),
                                       model_kwargs['y']['log_name'], out_list,
                                       t, model_kwargs['y']['log_id'])

        # End inpainting
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def apply_inpainting(self, x_t, model_kwargs):
        # Do inpainting here. Without adding noise to x_t
        inpainting_mask, inpainted_motion = model_kwargs['y'][
            'inpainting_mask'], model_kwargs['y']['inpainted_motion']
        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for now!'
        # assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape

        if self.data_transform_fn is not None:  #
            # model_output shape [1, 263, 1, 120]
            # Project back to motion representation
            unprojected_x_start = self.data_inv_transform_fn(
                x_t.permute(0, 2, 3, 1))  # [1, 1, 120, 263])
            # Inpaint
            painted_model_output = (
                (unprojected_x_start * ~inpainting_mask.permute(0, 2, 3, 1)) +
                (inpainted_motion.permute(0, 2, 3, 1) *
                 inpainting_mask.permute(0, 2, 3, 1)))
            # Do random projection again
            out_mean_new = self.data_transform_fn(
                painted_model_output).permute(0, 3, 1, 2)
        else:
            out_mean_new = (x_t * ~inpainting_mask) + (inpainted_motion *
                                                       inpainting_mask)

        return out_mean_new


    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(
                self.p_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    skip_timesteps=skip_timesteps,
                    init_image=init_image,
                    randomize_class=randomize_class,
                    cond_fn_with_grad=cond_fn_with_grad,
                    const_noise=const_noise,
                )):
            if dump_steps is not None and i in dump_steps:
                # dump.append(deepcopy(sample["sample"]))
                dump.append(deepcopy(sample["pred_xstart"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]


    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        #
        # skip_timesteps = 600
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device,
                           dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        previous_xstart = img

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0,
                                               high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                # NOTE: potential performance improvement for unconditional generation (we don't need grad)
                # when the generation is unconditional, we use p_sample
                if 'gmd' in model_kwargs['y'].keys(): # TODO: Remove
                    cond_fn_with_grad = cond_fn is not None
                    cond_fn_with_grad = True  # NOTE: for eval debugging
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                    previous_xstart=previous_xstart  # NOTE: for debugging
                )
                yield out
                img = out["sample"]
                previous_xstart = out["pred_xstart"]


    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        previous_xstart=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            previous_xstart=previous_xstart,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn,
                                       out_orig,
                                       x,
                                       t,
                                       model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"]
        }  # out_orig["pred_xstart"]} ####

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        previous_xstart=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                previous_xstart=previous_xstart,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn,
                                                     out_orig,
                                                     x,
                                                     t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {
            "sample": sample,
            "pred_xstart": out_orig["pred_xstart"].detach()
        }

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor(
                   self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t,
                                              x.shape)

        # Equation 12. reversed
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_next) +
                     th.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        # if dump_steps is not None:
        #     raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        if dump_steps is not None:
            dump = []

        final = None
        for i, sample in enumerate(
                self.ddim_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    eta=eta,
                    skip_timesteps=skip_timesteps,
                    init_image=init_image,
                    randomize_class=randomize_class,
                    cond_fn_with_grad=cond_fn_with_grad,
                )):
            # final = sample
            if dump_steps is not None and i in dump_steps:
                # dump.append(deepcopy(sample["sample"]))
                dump.append(deepcopy(sample["pred_xstart"]))
            final = sample
        if dump_steps is not None:
            return dump

        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device,
                           dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        # NOTE: for debugging
        previous_xstart = img

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0,
                                               high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                cond_fn_with_grad = True
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    previous_xstart=previous_xstart  # NOTE: for debugging
                )
                yield out
                img = out["sample"]
                previous_xstart = out["pred_xstart"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad
                                     and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(
                            cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn,
                                                   out_orig,
                                                   x,
                                                   t,
                                                   model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(
                1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(
                1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] +
                             5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] +
                             37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(
                1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 -
                                                                  nonzero_mask)

        return {
            "sample": sample,
            "pred_xstart": out_orig["pred_xstart"],
            "old_eps": old_eps
        }

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
                cond_fn_with_grad=cond_fn_with_grad,
                order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device,
                           dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0,
                                               high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(self,
                      model,
                      x_start,
                      x_t,
                      t,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model,
                                   x_t,
                                   t,
                                   clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"],
                       out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self,
                        model,
                        x_start,
                        t,
                        model_kwargs=None,
                        noise=None,
                        dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']

        def input_process(x):
            if self.conf.train_trajectory_only_xz:
                # [bs, njoints, nfeats, seqlen]
                return x[:, [1, 2]]
            else:
                return x

        # support training only xz
        # [bs, 4 or 263, 1, seqlen]
        x_start = input_process(x_start)

        enc = model.model
        mask = model_kwargs['y']['mask']
        get_xyz = lambda sample: enc.rot2xyz(
            sample,
            mask=None,
            pose_rep=enc.pose_rep,
            translation=enc.translation,
            glob=enc.glob,
            # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
            jointstype='smpl',  # 3.4 iter/sec
            vertstrans=False)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        if self.conf.apply_zero_mask:
            x_t = x_t * mask

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE: # Default
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if isinstance(model_output, tuple):
                # model has two heads
                # the other head would predict x0
                assert len(model_output) == 2
                assert self.model_mean_type == ModelMeanType.EPSILON
                model_output, model_output2 = model_output
            else:
                model_output2 = None

            if self.model_var_type in [
                    ModelVarType.LEARNED,
                    ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output,
                                                          C,
                                                          dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values],
                                    dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X:
                self.q_posterior_mean_variance(x_start=x_start, x_t=x_t,
                                               t=t)[0],
                ModelMeanType.START_X:
                x_start,
                ModelMeanType.EPSILON:
                noise,
            }[self.model_mean_type]

            # nfeats = 1
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            # overweighting the trajectory
            # weights for each features, multiples for the whole length of the sequence
            # [bs, njoints, nfeats, 1]
            weights = torch.ones(*target.shape[:-1],
                                 1,
                                 device=target.device,
                                 dtype=target.dtype)
            # power 2 because the weights are applies outside of the squared loss
            weights[:, :4] *= self.conf.traj_extra_weight**2

            time_weights = torch.ones(*target.shape,
                                       device=target.device,
                                       dtype=target.dtype)

            # Zero out the loss over the observed part of th motion
            if enc.zero_keyframe_loss:
                assert enc.keyframe_conditioned
                assert 'obs_mask' in model_kwargs.keys()
                assert torch.all(model_kwargs['obs_mask'] == (model_kwargs['obs_mask'] * mask).bool())
                mask = mask * (~ model_kwargs['obs_mask'])

            terms["rot_mse"] = self.masked_l2_weighted(
                target, model_output, mask,
                weights=weights,
                time_weights=time_weights)  # mean_flat(rot_mse)

            if model_output2 is not None:
                # model has two heads another loss function is x0
                target2 = x_start
                terms["rot_mse2"] = self.masked_l2_weighted(target2,
                                                            model_output2,
                                                            mask,
                                                            weights=weights,
                                                            time_weights=time_weights)
            if enc.keyframe_conditioned:
                # Compute the loss over the keyframes for logging
                obs_mask = mask * model_kwargs['obs_mask']
                terms["keyframes_mse"] = self.masked_l2_weighted(target, model_output, obs_mask, weights=weights, time_weights=time_weights, over_keyframes=True)

            target_xyz, model_output_xyz = None, None

            if self.lambda_rcxyz > 0.:
                assert self.conf.traj_extra_weight == 1
                target_xyz = get_xyz(
                    target
                )  # [bs, nvertices(vertices)/njoints(smpl), 3, nframes]
                model_output_xyz = get_xyz(
                    model_output)  # [bs, nvertices, 3, nframes]
                terms["rcxyz_mse"] = self.masked_l2(
                    target_xyz, model_output_xyz,
                    mask)  # mean_flat((target_xyz - model_output_xyz) ** 2)

            if self.lambda_vel_rcxyz > 0.:
                assert self.conf.traj_extra_weight == 1
                if self.data_rep == 'rot6d' and dataset.dataname in [
                        'humanact12', 'uestc'
                ]:
                    target_xyz = get_xyz(
                        target) if target_xyz is None else target_xyz
                    model_output_xyz = get_xyz(
                        model_output
                    ) if model_output_xyz is None else model_output_xyz
                    target_xyz_vel = (target_xyz[:, :, :, 1:] -
                                      target_xyz[:, :, :, :-1])
                    model_output_xyz_vel = (model_output_xyz[:, :, :, 1:] -
                                            model_output_xyz[:, :, :, :-1])
                    terms["vel_xyz_mse"] = self.masked_l2(
                        target_xyz_vel, model_output_xyz_vel, mask[:, :, :,
                                                                   1:])

            if self.lambda_fc > 0.:
                assert self.conf.traj_extra_weight == 1
                torch.autograd.set_detect_anomaly(True)
                if self.data_rep == 'rot6d' and dataset.dataname in [
                        'humanact12', 'uestc'
                ]:
                    target_xyz = get_xyz(
                        target) if target_xyz is None else target_xyz
                    model_output_xyz = get_xyz(
                        model_output
                    ) if model_output_xyz is None else model_output_xyz
                    # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11
                    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
                    relevant_joints = [
                        l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx
                    ]
                    gt_joint_xyz = target_xyz[:,
                                              relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                    gt_joint_vel = torch.linalg.norm(
                        gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1],
                        axis=2)  # [BatchSize, 4, Frames]
                    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01),
                                              dim=2).repeat(1, 1, 3, 1)
                    pred_joint_xyz = model_output_xyz[:,
                                                      relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                    pred_vel = pred_joint_xyz[:, :, :,
                                              1:] - pred_joint_xyz[:, :, :, :-1]
                    pred_vel[~fc_mask] = 0
                    terms["fc"] = self.masked_l2(
                        pred_vel,
                        torch.zeros(pred_vel.shape, device=pred_vel.device),
                        mask[:, :, :, 1:])
            if self.lambda_vel > 0.:
                assert self.conf.traj_extra_weight == 1
                target_vel = (target[..., 1:] - target[..., :-1])
                model_output_vel = (model_output[..., 1:] -
                                    model_output[..., :-1])
                terms["vel_mse"] = self.masked_l2(
                    target_vel[:, :
                               -1, :, :],  # Remove last joint, is the root location!
                    model_output_vel[:, :-1, :, :],
                    mask[:, :, :, 1:]
                )  # mean_flat((target_vel - model_output_vel) ** 2)

            terms["loss"] = terms["rot_mse"] + terms.get('rot_mse2', 0.) + terms.get('vb', 0.) +\
                            (self.lambda_vel * terms.get('vel_mse', 0.)) +\
                            (self.lambda_rcxyz * terms.get('rcxyz_mse', 0.)) + \
                            (self.lambda_fc * terms.get('fc', 0.))

            if self.conf.time_weighted_loss:
                # time weighted the loss function so that the epsilon-based loss would pay more attention to T ~ 1000
                time_weights = _extract_into_tensor(self.ratio_eps, t,
                                                    terms['loss'].shape)
                time_weights /= time_weights.mean()
                terms['loss'] = terms['loss'] * time_weights

            if self.conf.train_x0_as_eps:
                time_weights = _extract_into_tensor(
                    self.sqrt_alphas_cumprod_over_oneminus_aphas_cumprod, t,
                    terms['loss'].shape)
                time_weights /= time_weights.mean()
                terms['loss'] = terms['loss'] * time_weights

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()

        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11

        gt_joint_xyz = gt_xyz[:, [
            l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx
        ], :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] -
                                         gt_joint_xyz[:, :, :, :-1],
                                         axis=2)  # [BatchSize, 4, Frames]
        fc_mask = (gt_joint_vel <= 0.01)
        pred_joint_xyz = pred_xyz[:, [
            l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx
        ], :, :]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] -
                                           pred_joint_xyz[:, :, :, :-1],
                                           axis=2)  # [BatchSize, 4, Frames]
        pred_joint_vel[
            ~fc_mask] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)
        return self.masked_l2(
            pred_joint_vel,
            torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
            mask[:, :, :, 1:])


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size,
                      device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self,
                      model,
                      x_start,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start)**2))
            eps = self._predict_eps_from_xstart(x_t, t_batch,
                                                out["pred_xstart"])
            mse.append(mean_flat((eps - noise)**2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)