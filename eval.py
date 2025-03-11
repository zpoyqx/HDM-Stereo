import argparse
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Tool.evaluation import evaluate, evaluate_multi
from Tool.model import StereoModel
from Tool.optimize import translate_optimize
from dataset import StereoDataset


class Evaluation:
    def __init__(self, config):
        self.threshold = config.threshold
        self.device = torch.device('cuda', 0)
        self.model = StereoModel(config.checkpoint_path, isprint=False)
        config.model_config = self.model.model_config
        self.batch_size = config.batch_size
        self.val_loader = DataLoader(
            dataset=StereoDataset(config, False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        ranges = {'US3D': (-96, 96), 'WHU-Stereo': (-128, 64), 'WHU-MVS': (0, 168)}
        self.range = ranges.get(config.dataset, None)

    def to_cuda(self, tensors):
        if isinstance(tensors, (list, tuple)):
            return [tensor.cuda(self.device, non_blocking=True) for tensor in tensors]
        else:
            return tensors.cuda(self.device, non_blocking=True)

    @torch.no_grad()
    def val(self, optimize=False, err_dist=False):
        epe, bad, total_time = 0, 0, 0
        err, num = {}, {}
        length = len(self.val_loader)

        for inputs, target in tqdm(self.val_loader, file=sys.stdout, smoothing=0, ncols=80):
            inputs = self.to_cuda(inputs)
            target = self.to_cuda(target)
            output = translate_optimize(self.model, *inputs, isprint=False) if optimize else self.model(*inputs)

            errors = evaluate(output, target, self.threshold, self.range)
            epe += errors[0]
            bad += errors[1]

            if err_dist:
                output, target = output.flatten().cpu().numpy(), target.int().flatten().cpu().numpy()
                mask = np.logical_and(target >= self.range[0], target <= self.range[1])  # 使用掩码来忽略无效值
                valid_output = output[mask]
                valid_target = target[mask]
                diff = np.abs(valid_output - valid_target)

                # 获取唯一的 target 值及其索引
                unique_targets, inverse_indices = np.unique(valid_target, return_inverse=True)

                # 按照每个唯一 target 值累加误差和计数
                for i, key in enumerate(unique_targets):
                    err[key] = err.get(key, 0) + diff[inverse_indices == i].sum()
                    num[key] = num.get(key, 0) + np.sum(inverse_indices == i)

        print(f'EPE: {epe / length:.2f}\nBAD{self.threshold}: {bad / length * 100:.2f} %')
        if err_dist:
            result = {key: err[key] / num[key] for key in err}
            result = sorted(result.items(), key=lambda x: x[0])
            result = [[key, value] for key, value in result]  # 输出json格式
            print(result)

    @torch.no_grad()
    def val_large_disp(self, optimize=False):
        intervals = [50, 100, 168]
        epe, bad, total_pixels = np.zeros(len(intervals)), np.zeros(len(intervals)), np.zeros(len(intervals))
        for inputs, target in tqdm(self.val_loader, file=sys.stdout, smoothing=0, ncols=80):
            inputs = self.to_cuda(inputs)
            target = self.to_cuda(target)
            output = translate_optimize(self.model, *inputs, isprint=False) if optimize else self.model(*inputs)
            errors = evaluate_multi(output, target, intervals, self.threshold)
            epe += errors[0]
            bad += errors[1]
            total_pixels += errors[2]
        for i, interval in enumerate(intervals):
            print(f'范围:[0, {interval}]')
            print(f'EPE: {epe[i] / total_pixels[i]:.2f}\nBAD{self.threshold}: {bad[i] / total_pixels[i] * 100:.2f} %')

    @torch.no_grad()
    def val_speed(self, optimize=False):
        for inputs, _ in tqdm(self.val_loader, file=sys.stdout, smoothing=0, ncols=80):
            inputs = self.to_cuda(inputs)
            translate_optimize(self.model, *inputs, isprint=False) if optimize else self.model(*inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data-path', type=str, default='data', help='测试集路径(包含val文件夹)')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='使用的数据集')
    parser.add_argument('-c', '--checkpoint-path', type=str, required=True, help='权重路径')
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-t', '--threshold', type=int, default=3, help='错误像素的判断阈值')
    parser.add_argument('-n', '--num-workers', type=int, default=0)
    parser.add_argument('-o', '--optimize', action='store_true', help='是否使用HDM-Stereo')
    config = parser.parse_args()

    assert osp.exists(config.checkpoint_path), '权重路径不存在'
    assert osp.exists(config.data_path), '数据集路径不存在'

    evaluator = Evaluation(config)
    evaluator.val(optimize=config.optimize)
