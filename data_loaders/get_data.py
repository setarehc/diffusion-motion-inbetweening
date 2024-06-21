from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, amass_collate
from typing import Tuple
from dataclasses import dataclass


def get_dataset_class(name):
    if name == "amass":
        from data_loaders.amass.data.dataset import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name == 'amass':
        return amass_collate
    else:
        return all_collate


@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_frames: int
    split: str = 'train'
    hml_mode: str = 'train'
    use_abs3d: bool = False
    traject_only: bool = False
    use_random_projection: bool = False
    random_projection_scale: float = None
    augment_type: str = 'none'
    std_scale_shift: Tuple[float] = (1.0, 0.0)
    drop_redundant: bool = False


def get_dataset(conf: DatasetConfig):
    DATA = get_dataset_class(conf.name)
    if conf.name in ["humanml", "kit"]:
        dataset = DATA(split=conf.split,
                       num_frames=conf.num_frames,
                       mode=conf.hml_mode,
                       use_abs3d=conf.use_abs3d,
                       traject_only=conf.traject_only,
                       use_random_projection=conf.use_random_projection,
                       random_projection_scale=conf.random_projection_scale,
                       augment_type=conf.augment_type,
                       std_scale_shift=conf.std_scale_shift,
                       drop_redundant=conf.drop_redundant)
    elif conf.name == "amass":
        dataset = DATA(split=conf.split)
    else:
        raise NotImplementedError()
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(conf: DatasetConfig, shuffle=True, num_workers=8, drop_last=True):
    dataset = get_dataset(conf)
    collate = get_collate_fn(conf.name, conf.hml_mode)

    # return dataset
    loader = DataLoader(dataset,
                        batch_size=conf.batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        collate_fn=collate,)
                        #pin_memory=True) # Remove if out of memory occurs

    return loader