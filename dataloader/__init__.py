from .v2_dataset import V2Dataset, get_v2_pallete
from .s4_dataset import S4Dataset
from .ms3_dataset import MS3Dataset
# from .synthetic_dataset import SyntheticDataset
# from mmcv import Config


def build_dataset(type, split, args):
    if type == 'V2Dataset':
        return V2Dataset(split=split, cfg=args)
    elif type == 'S4Dataset':
        return S4Dataset(split=split, cfg=args)
    elif type == 'MS3Dataset':
        return MS3Dataset(split=split, cfg=args)
    # elif type == 'SyntheticDataset':
        # return SyntheticDataset(split=split, cfg=args)
    else:
        raise ValueError


__all__ = ['build_dataset', 'get_v2_pallete']
