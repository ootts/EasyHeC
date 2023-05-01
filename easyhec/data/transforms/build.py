from . import transforms as T
from .transforms import ClipRange, CenterCrop


def build_transforms(cfg, is_train=True):
    ts = []
    for transform in cfg.input.transforms:
        ts.append(build_transform(transform))
    transform = T.Compose(ts)
    return transform


def build_transform(t):
    if t['name'] == 'ClipRange':
        return ClipRange(t['range'])
    elif t['name'] == 'CenterCrop':
        return CenterCrop(t['width'], t['height'])
    else:
        raise NotImplementedError()
