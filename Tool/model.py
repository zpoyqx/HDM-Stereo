import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Union

import torch
from torch import nn

from HMSMNet import HMSMNet
from Module.computation import Estimation
from Module.cost import CostVolume
from PSMNet import PSMNet
from StereoNet import StereoNet


def model_parameter(parser: ArgumentParser, config: Union[Namespace, dict], args=()):
    config = Namespace(**config) if isinstance(config, dict) else config
    if config.dataset == 'US3D':
        parser.add_argument('--in-channel', type=int, default=1, help='通道数量')
        parser.add_argument('--min-disp', type=int, default=-96, help='最小视差')
        parser.add_argument('--max-disp', type=int, default=96, help='最大视差')
    elif config.dataset == 'WHU-Stereo':
        parser.add_argument('--in-channel', type=int, default=1, help='通道数量')
        parser.add_argument('--min-disp', type=int, default=-128, help='最小视差')
        parser.add_argument('--max-disp', type=int, default=64, help='最大视差')
    elif config.dataset == 'WHU-MVS':
        parser.add_argument('--in-channel', type=int, default=1, help='通道数量')
        parser.add_argument('--min-disp', type=int, default=0, help='最小视差')
        parser.add_argument('--max-disp', type=int, default=255, help='最大视差')
    else:
        raise NotImplementedError(f'数据集({config.dataset})未实现')

    return parser.parse_args(args)


def build_model(checkpoint_path, extra_config=None, return_config=False):
    model_name = osp.basename(checkpoint_path).split('-')[0]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model_config = checkpoint.get('model_config')
    model_config.__dict__.update(extra_config or {})

    if model_name == 'hmsm':
        model = HMSMNet(model_config)
    elif model_name == 'stereo':
        model = StereoNet(model_config)
    elif model_name == 'psm':
        model = PSMNet(model_config)
    else:
        raise NotImplementedError
    model.to('cuda:0')
    model = nn.parallel.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    if return_config:
        return model, model_config
    else:
        return model


class StereoModel:
    # 该类用于快速代价体裁剪
    def __init__(self, checkpoint_path, iscrop=True, extra_config=None, isprint=False):
        self.model_name = osp.basename(checkpoint_path).split('-')[0]
        assert self.model_name in ['hmsm', 'stereo', 'psm']
        self.net, self.model_config = build_model(checkpoint_path, extra_config, return_config=True)
        self.model_min_disp = self.net.module.min_disp
        self.model_max_disp = self.net.module.max_disp
        self.min_disp = self.model_min_disp
        self.max_disp = self.model_max_disp
        self.method = getattr(self.net.module, 'method', 'concat')
        self.iscrop = iscrop
        self.isprint = isprint

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def crop_costvolume(self, min_disp, max_disp):
        # 如果使用了代价体裁剪，请在每次推理前都调用该函数
        # CostVolume和Estimation没有参数，因此不需要重新加载权重
        if not self.iscrop:
            return
        assert min_disp < max_disp, 'min_disp必须小于max_disp'
        min_disp, max_disp = (min_disp // 16 - 1) * 16, (max_disp // 16 + 2) * 16
        min_disp, max_disp = max(min_disp, self.model_min_disp), min(max_disp, self.model_max_disp)
        if self.min_disp == min_disp and self.max_disp == max_disp:
            return
        self.min_disp, self.max_disp = min_disp, max_disp
        if self.isprint:
            print(f'限制范围:[{min_disp}, {max_disp}]')

        if self.model_name == 'hmsm':
            self.net.module.cost0 = CostVolume(min_disp=min_disp // 4, max_disp=max_disp // 4, method=self.method)
            self.net.module.cost1 = CostVolume(min_disp=min_disp // 8, max_disp=max_disp // 8, method=self.method)
            self.net.module.cost2 = CostVolume(min_disp=min_disp // 16, max_disp=max_disp // 16, method=self.method)
            self.net.module.estimator0 = Estimation(min_disp=min_disp // 4, max_disp=max_disp // 4)
            self.net.module.estimator1 = Estimation(min_disp=min_disp // 8, max_disp=max_disp // 8)
            self.net.module.estimator2 = Estimation(min_disp=min_disp // 16, max_disp=max_disp // 16)
        elif self.model_name == 'stereo':
            self.net.module.cost = CostVolume(min_disp=min_disp // 8, max_disp=max_disp // 8, method=self.method)
            self.net.module.computer = Estimation(min_disp=min_disp // 8, max_disp=max_disp // 8)
        elif self.model_name == 'psm':
            self.net.module.disp_length = int(max_disp) - int(min_disp)
            self.net.module.cost = CostVolume(min_disp=min_disp // 4, max_disp=max_disp // 4, method=self.method)
            self.net.module.computer = Estimation(min_disp=min_disp, max_disp=max_disp)
