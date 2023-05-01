import random

import numpy as np
import torch
import torchvision
from dl_ext.primitive import safe_zip
from dl_ext.timer import EvalTime
from torchvision.transforms import functional as F

from easyhec.utils.pn_utils import stack


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, target=None, **kwargs):
        if target is None:
            for t in self.transforms:
                inputs = t(inputs, **kwargs)
            return inputs
        else:
            for t in self.transforms:
                inputs, target = t(inputs, target, **kwargs)
            return inputs, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

    def get_voxel_sizes(self):
        for t in self.transforms:
            if hasattr(t, 'voxel_sizes'):
                return getattr(t, 'voxel_sizes')
        return None


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, image, target=None, **kwargs):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        if hasattr(target, 'resize'):
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None, **kwargs):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None and hasattr(target, 'transpose'):
                target = target.transpose(0)
        if target is None:
            return image
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target=None, **kwargs):
        image = self.color_jitter(image)
        if target is None:
            return image
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None, **kwargs):
        image = F.to_tensor(image)
        if target is None:
            return image
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, **kwargs):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class ClipRange(object):
    def __init__(self, range):
        """
        Keep point cloud only in specified range.
        :param range: xmin,ymin,zmin,xmax,ymax,zmax
        """
        self.range = range

    def __call__(self, inputs, targets=None, **kwargs):
        range = kwargs.get('range', self.range)
        coord0 = inputs['sample0']['coord']
        keep0 = np.logical_and.reduce([
            coord0[:, 0] > range[0],
            coord0[:, 1] > range[1],
            coord0[:, 2] > range[2],
            coord0[:, 0] < range[3],
            coord0[:, 1] < range[4],
            coord0[:, 2] < range[5]
        ])
        inputs['sample0']['coord'] = inputs['sample0']['coord'][keep0]
        inputs['sample0']['color'] = inputs['sample0']['color'][keep0]
        inputs['sample0']['pixelloc_to_voxelidx'].fill(-1)
        inputs['sample0']['pixelloc_to_voxelidx'].reshape(-1)[keep0.nonzero()[0]] = np.arange(
            keep0.nonzero()[0].shape[0])
        if targets is not None:
            targets['flow3d'] = targets['flow3d'][keep0]
        coord1 = inputs['sample1']['coord']
        keep1 = np.logical_and.reduce([
            coord1[:, 0] > range[0],
            coord1[:, 1] > range[1],
            coord1[:, 2] > range[2],
            coord1[:, 0] < range[3],
            coord1[:, 1] < range[4],
            coord1[:, 2] < range[5]
        ])
        inputs['sample1']['coord'] = inputs['sample1']['coord'][keep1]
        inputs['sample1']['color'] = inputs['sample1']['color'][keep1]
        inputs['sample1']['pixelloc_to_voxelidx'].fill(-1)
        inputs['sample1']['pixelloc_to_voxelidx'].reshape(-1)[keep1.nonzero()[0]] = np.arange(
            keep1.nonzero()[0].shape[0])

        if targets is None:
            return inputs
        else:
            return inputs, targets


class CenterCrop:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, inputs, targets=None, **kwargs):
        oh, ow = inputs['image'].shape[:2]
        ch = oh // 2
        cw = ow // 2
        sh = ch - self.height // 2
        eh = sh + self.height
        sw = cw - self.width // 2
        ew = sw + self.width
        inputs['image'] = inputs['image'][sh:eh, sw:ew]
        inputs['depth'] = inputs['depth'][sh:eh, sw:ew]
        inputs['mask'] = inputs['mask'][sh:eh, sw:ew]
        inputs['forward_flow'] = inputs['forward_flow'][sh:eh, sw:ew]
        inputs['backward_flow'] = inputs['backward_flow'][sh:eh, sw:ew]
        inputs['H'] = self.height
        inputs['W'] = self.width
        inputs['K'][0, 2] = inputs['K'][0, 2] - sw
        inputs['K'][1, 2] = inputs['K'][1, 2] - sh
        return inputs
