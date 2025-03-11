import argparse
import os
import os.path as osp
import random
import sys
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from Tool.model import model_parameter
from dataset import StereoDataset
from solver import Solver


def set_seed(test_status, seed):
    if test_status:
        np.random.seed(seed)  # 固定numpy随机种子
        random.seed(seed)  # 固定random随机种子
        os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

        torch.backends.cudnn.benchmark = False  # 禁用卷积优化
        torch.manual_seed(seed)  # 固定CPU随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 固定当前GPU随机种子
            torch.cuda.manual_seed_all(seed)  # 固定所有GPU随机种子
            # 操作均使用确定性算法，当无确定性算法可用时，产生警报
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True  # (适用于网络结构固定、输入形状固定)启用卷积优化


def train(config, distributed):
    train_data, val_data = StereoDataset(config, True), StereoDataset(config, False)
    if distributed:
        dist.init_process_group(backend='nccl')
        set_seed(config.test, config.seed + dist.get_rank())  # 给每个进程设置不同的随机种子
        # DistributedSampler只能在分布式训练中使用，每个GPU只训练数据集的一个子集
        train_sampler, val_sampler = DistributedSampler(train_data), DistributedSampler(val_data)
    else:
        set_seed(config.test, config.seed)
        train_sampler, val_sampler = None, None
    # 如果显存不够时可以考虑pin_memory=False
    train_loader = DataLoader(
        dataset=train_data,
        shuffle=not distributed,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=max(config.batch_size // 4, 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )

    solver = Solver(config, train_loader, val_loader)
    solver.train()


def train_without_val(config, distributed):
    train_data = StereoDataset(config, True)
    if distributed:
        dist.init_process_group(backend='nccl')
        set_seed(config.test, config.seed + dist.get_rank())  # 给每个进程设置不同的随机种子
        # DistributedSampler只能在分布式训练中使用，每个GPU只训练数据集的一个子集
        train_sampler = DistributedSampler(train_data)
    else:
        set_seed(config.test, config.seed)
        train_sampler = None
    # 如果显存不过可以考虑pin_memory=False
    train_loader = DataLoader(
        dataset=train_data,
        shuffle=not distributed,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    solver = Solver(config, train_loader, None)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    # 数据集参数
    parser.add_argument(
        '-ds', '--dataset', type=str, choices=['US3D', 'WHU-Stereo', 'WHU-MVS'], required=True, help='使用的数据集'
    )
    parser.add_argument('-p', '--data-path', type=str, default='data', help='使用的数据集路径')
    parser.add_argument('-a', '--aug', action='store_true', help='是否进行数据增强')
    parser.add_argument('-l', '--limit', action='store_true', help='是否限制裁剪不出现黑边')

    # 训练参数
    parser.add_argument('-m', '--model', type=str, choices=['hmsm', 'stereo', 'psm'], required=True, help='使用的网络')
    parser.add_argument('-e', '--max-epoch', type=int, default=60)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-c', '--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-d', '--dist', action='store_true', help='是否进行分布式训练', dest='dist_status')
    parser.add_argument('-t', '--test', action='store_true', help='是否固定随机数种子使训练可复现')
    parser.add_argument('-v', '--verbose', action='store_true', help='是否打印进度条')

    # 验证参数
    parser.add_argument('--val-step', type=int, default=1, help='验证步长')
    parser.add_argument('--threshold', type=int, default=3, help='错误像素的判断阈值')

    # 数据记录
    parser.add_argument('-w', '--wandb', action='store_true', help='是否使用wandb', dest='wandb_status')
    parser.add_argument('-o', '--offline', action='store_true', help='是否离线运行wandb')
    parser.add_argument('-n', '--note', type=str, help='wandb训练备注')

    # 保存和重载权重
    parser.add_argument('-s', '--save', action='store_true', help='是否保存最优权重', dest='save_status')
    parser.add_argument('--val-metric', type=str, default='avgerr', help='最优权重的评判指标')
    parser.add_argument('-r', '--restore-status', type=str, choices=['all', 'weight_only'], help='权重加载')
    parser.add_argument('-rp', '--restore-path', type=str, help='加载权重的路径')
    parser.add_argument('-i', '--id', type=str, help='断点恢复时需要指定的wandb.run.id')

    # 其他参数
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--locals', action='store_true', help='出错打印栈回溯信息时是否显示变量')

    # 模型超参数，设置需要指定params(必须放在最后)
    # 示例：python train.py -ws -m hmsm -ds US3D params --min-disp -96 --max-disp 96
    try:
        argv = sys.argv[1:]
        idx = argv.index('params')
        args = argv[idx + 1 :]
        config = parser.parse_args(argv[:idx])
    except ValueError:
        config = parser.parse_args()
        args = []
    model_parser = sub_parsers.add_parser('params')
    model_config = model_parameter(model_parser, config, args)
    config.model_config = model_config

    # 预处理
    config.seed = config.seed or int(time())
    config.default_save_dir = 'checkpoints'
    config.data_path = osp.join(config.data_path, config.dataset)
    assert config.wandb_status or osp.exists(config.default_save_dir), f'默认保存路径不存在：{config.defaut_save_dir}'
    assert config.restore_status is None or osp.exists(config.restore_path), f'权重路径不存在：{config.restore_path}'
    assert not config.offline or config.wandb_status, '使用wandb离线模式时，必须指定-w/--wandb参数'
    assert torch.cuda.is_available(), '无法进行GPU训练'
    if config.dist_status:
        assert torch.cuda.device_count() > 1, '分布式训练时可用GPU数量至少要大于1个'
        assert dist.is_available() and dist.is_nccl_available(), '不满足分布式训练的条件'
    if config.val_step == 0 or config.val_step >= config.max_epoch:
        train_without_val(config, distributed=config.dist_status)
    else:
        train(config, distributed=config.dist_status)
