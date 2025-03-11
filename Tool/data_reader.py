from random import randint, random

import numpy as np
import skimage
import torch
from torchvision.transforms.v2 import Compose, RandomCrop, ToDtype, ToImage
from torchvision.transforms.v2.functional import crop


def to_cuda(*data, device='cuda:0', unsqueeze=True, pre_fn=None, post_fn=None):
    result = []
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    for item in data:
        if pre_fn is not None:
            item = pre_fn(item)
        if not isinstance(item, torch.Tensor):
            item = transform(np.expand_dims(item, -1)) if item.ndim == 2 else transform(item)
        if post_fn is not None:
            item = post_fn(item)
        item = item.to(device, non_blocking=True)
        result.append(item.unsqueeze(0) if unsqueeze else item)
    return result if len(result) > 1 else result[0]


def to_numpy(*data, squeeze=True):
    result = []
    for item in data:
        if item is None:
            result.append(None)
        else:
            item = item.cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            result.append(np.squeeze(item) if squeeze else item)
    return result if len(result) > 1 else result[0]


def RTT(left, right, disp, size=512, limit=False):
    size = (size, size) if isinstance(size, int) else size
    left_params = list(RandomCrop.get_params(left, output_size=size))
    right_params = left_params[:]
    offset = 0
    if random() < 0.7:
        # limit为True：阻止平移超过边界
        # limit为False：如果平移后裁剪区域超过图像边界，以黑色填充
        offset = randint(-100, 100)
        if limit and (left_params[1] + offset < 0 or left_params[1] + offset > left.shape[-1] - size[1]):
            w = left.shape[-1]
            offset = offset if size[1] - w < offset < w - size[1] else 0
            left_params[1] = randint(max(-offset, 0), min(w - size[1], w - size[1] - offset))
        right_params[1] = left_params[1] + offset
    left = crop(left, *left_params)
    right = crop(right, *right_params)
    disp = crop(disp, *left_params)
    # 剪切图向右平移相当于坐标向左平移
    disp = disp.where(disp.eq(-999.0), disp + offset)
    return (left, right), disp


class ImageReader:
    def __init__(self, dataset, in_channel, train=True, aug=True, limit=False):
        self.dataset = dataset
        self.as_gray = in_channel == 1
        self.train = train
        self.size = 384 if self.dataset == 'WHU-MVS' else 512
        self.rc = RandomCrop(self.size)
        self.totensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        self.aug = aug
        self.limit = limit

    def read_image(self, file):
        image = skimage.io.imread(file, as_gray=self.as_gray).astype('float32')
        image = np.expand_dims(image, -1) if self.as_gray else image
        image = (image - np.mean(image)) / np.std(image)
        return self.totensor(image)

    def read_disp(self, file):
        image = skimage.io.imread(file).astype('float32')
        image = np.expand_dims(image, -1)
        image = image / 256.0 if self.dataset == 'WHU-MVS' else image
        return self.totensor(image)

    def read_eval(self, left_file, right_file):
        left = self.read_image(left_file)
        right = self.read_image(right_file)
        return left, right

    def read(self, left_file, right_file, disp_file):
        left = self.read_image(left_file)
        right = self.read_image(right_file)
        disp = self.read_disp(disp_file)
        if self.train:
            if self.aug:
                return RTT(left, right, disp, self.size, self.limit)
            else:
                params = RandomCrop.get_params(left, output_size=(self.size, self.size))
                return (crop(left, *params), crop(right, *params)), crop(disp, *params)
        else:
            return (left, right), disp
