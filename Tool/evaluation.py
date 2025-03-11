from contextlib import contextmanager
from timeit import default_timer

import cv2
import numpy as np
import torch

from Tool.data_reader import to_numpy


def myprint(msg, isprint=True):
    if isprint:
        print(msg)


def evaluate(outputs, targets, threshold, disp_range=None, masked=False):
    assert outputs.shape == targets.shape, '用于评估损失的两幅图像的尺寸必须相同'
    assert len(outputs.shape) == 4, '输入图像的shape必须为[B,C,H,W]'
    epe, bad, total_pixels = 0, 0, 0
    for output, target in zip(outputs, targets):
        if disp_range is None:
            mask = target != -999.0
        else:
            # 这个范围本身就不包含-999.0
            mask = torch.logical_and(target >= disp_range[0], target <= disp_range[1])
        diff = torch.abs(output - target)[mask]
        epe += diff.sum().item()
        bad += (diff > threshold).sum().item()
        total_pixels += diff.numel() if masked else target.numel()
    return epe / total_pixels, bad / total_pixels


def evaluate_np(output, target, mask=None, threshold=3, isprint=True):
    output, target, mask = to_numpy(output, target, mask)
    assert output.shape == target.shape, '用于评估损失的两幅图像的尺寸必须相同'
    mask = np.ones_like(target).astype(bool) if mask is None else mask
    assert mask.dtype == np.bool_, 'mask必须为bool类型'

    diff = np.abs(output - target)[mask]
    valid = diff.size / target.size
    epe, bad = diff.mean(), np.sum(diff > threshold) / diff.size
    myprint(f'有效像素比例: {valid * 100:.2f}%', isprint)
    myprint(f'EPE: {epe:.2f}\nBAD: {bad * 100:.2f} %', isprint)
    return epe, bad


def evaluate_multi(outputs, targets, intervals, threshold=3, masked=False):
    assert outputs.shape == targets.shape, '用于评估损失的两幅图像的尺寸必须相同'
    assert isinstance(intervals, (list, tuple)), 'intervals必须可迭代'
    epe, bad, total_pixels = np.zeros(len(intervals)), np.zeros(len(intervals)), np.zeros(len(intervals))
    for output, target in zip(outputs, targets):
        masks = [torch.abs(target) < interval for interval in intervals]
        diff = torch.abs(output - target)
        for i, mask in enumerate(masks):
            d = diff[mask]
            epe[i] += d.sum().item()
            bad[i] += (d > threshold).sum().item()
            total_pixels[i] += d.numel() if masked else target.numel()
    return epe, bad, total_pixels


def dye(image, colormap=cv2.COLORMAP_VIRIDIS):
    image = image.cpu().numpy().squeeze() if isinstance(image, torch.Tensor) else image
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return cv2.applyColorMap(image, colormap)


def gen_diff(output, target, mask=None, threshold=6, colormap=cv2.COLORMAP_VIRIDIS):
    output, target, mask = to_numpy(output, target, mask)
    assert output.shape == target.shape, '用于生成差异伪彩色图的两幅图像的尺寸必须相同'
    mask = np.ones_like(target).astype(bool) if mask is None else mask
    assert mask.dtype == np.bool_, 'mask必须为bool类型'

    diff = np.abs(np.where(mask, output - target, 0))  # mask为0的地方不计算差异
    diff_tmp = np.where(diff > threshold, threshold, diff)
    diff_norm = (diff_tmp - diff_tmp.min()) / (diff_tmp.max() - diff_tmp.min())
    diff_norm = (diff_norm * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, colormap)
    return diff, diff_color


def compare(src_before, src_after, target, mask=None, threshold=3, isprint=True):
    src_before, src_after, target, mask = to_numpy(src_before, src_after, target, mask)
    myprint('-----处理前-----', isprint)
    evaluate_np(src_before, target, mask=mask, threshold=threshold, isprint=isprint)
    diff_before, diff_color_before = gen_diff(src_before, target, mask=mask, threshold=threshold * 2)

    myprint('-----处理后-----', isprint)
    evaluate_np(src_after, target, mask=mask, threshold=threshold, isprint=isprint)
    diff_after, diff_color_after = gen_diff(src_after, target, mask=mask, threshold=threshold * 2)

    return diff_before, diff_after, diff_color_before, diff_color_after

