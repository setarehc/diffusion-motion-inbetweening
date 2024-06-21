from dataclasses import dataclass
from typing import Tuple
from utils.parser_util import DataOptions, ModelOptions, DiffusionOptions, TrainingOptions, EvaluationOptions


@dataclass
class _motion(ModelOptions, DataOptions, DiffusionOptions, TrainingOptions,
              EvaluationOptions):
    num_frames: int = 196
    predict_xstart: bool = True
    grad_clip: float = 1.
    avg_model_beta: float = 0.9999


@dataclass
class _traj(ModelOptions, DataOptions, DiffusionOptions, TrainingOptions,
            EvaluationOptions):
    num_frames: int = 196
    predict_xstart: bool = False
    grad_clip: float = 1.
    avg_model_beta: float = 0.9999
    batch_size: int = 64
    save_interval: int = 12_500
    num_steps: int = 100_000


@dataclass
class _motion_unet(_motion):
    # all UNETs use 224 as the training max length
    num_frames: int = 224
    weight_decay: float = 0.01
    use_fp16: bool = True
    arch: str = 'unet'
    latent_dim: int = 512
    unet_adagn: bool = True
    unet_zero: bool = True


@dataclass
class _traj_unet(_traj):
    # all UNETs use 224 as the training max length
    num_frames: int = 224
    weight_decay: float = 0.01
    use_fp16: bool = True
    arch: str = 'unet'
    latent_dim: int = 512
    unet_adagn: bool = True
    unet_zero: bool = True


@dataclass
class motion_mdm(_motion):
    arch: str = 'trans_enc'
    latent_dim: int = 512
    ff_size: int = 1024
    weight_decay: float = 0
    eval_use_avg: bool = False  # MDM doesn't use avg model during inference


@dataclass
class traj_mdm(_traj):
    pass


@dataclass
class motion_unet_adagn_xl(_motion_unet):
    dim_mults: Tuple[float] = (2, 2, 2, 2)


@dataclass
class motion_unet_adagn_xl_loss2(_motion_unet):
    dim_mults: Tuple[float] = (2, 2, 2, 2)
    traj_extra_weight: float = 2


@dataclass
class motion_unet_adagn_xl_loss5(_motion_unet):
    dim_mults: Tuple[float] = (2, 2, 2, 2)
    traj_extra_weight: float = 5


@dataclass
class motion_unet_adagn_xl_loss10(_motion_unet):
    dim_mults: Tuple[float] = (2, 2, 2, 2)
    traj_extra_weight: float = 10


@dataclass
class traj_unet_adagn_swx(_traj_unet):
    dim_mults: Tuple[float] = (0.125, 0.25, 0.5)


@dataclass
class traj_unet_xxs(_traj_unet):
    dim_mults: Tuple[float] = (0.0625, 0.125, 0.25, 0.5)
    unet_adagn: bool = False
    unet_zero: bool = False
