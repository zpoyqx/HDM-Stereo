import argparse

import torch
import torch.nn.functional as F
from torch import nn

from Module.computation import Estimation
from Module.cost import CostVolume
from .aggregation import FeatureFusion
from .feature import FeatureExtraction
from .refinement import Refinement


class StereoNet(nn.Module):
    def __init__(self, args: argparse.Namespace, method='concat'):
        super().__init__()
        self.in_channel = args.in_channel
        self.channel = 32
        self.min_disp = int(args.min_disp)
        self.max_disp = int(args.max_disp)
        self.method = method

        self.extractor = FeatureExtraction(self.in_channel, self.channel)
        self.cost = CostVolume(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, method=self.method)
        self.aggregator = FeatureFusion(self.channel * 2, self.channel)
        self.computer = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.refiner1 = Refinement(self.in_channel + 1, self.channel)
        self.refiner2 = Refinement(self.in_channel + 1, self.channel)
        self.refiner3 = Refinement(self.in_channel + 1, self.channel)

        self._initialize_weights()

    def forward(self, left_image, right_image):
        """
        Args:
            left_image: 左图像
            right_image: 右图像

        Returns:
            list: [disparity]
        """

        # [N, C, H/8, W/8]
        left_feature = self.extractor(left_image)
        right_feature = self.extractor(right_image)
        # [N, 2*C, H/8, W/8, D]
        cost_volume = self.cost(left_feature, right_feature)
        # [N, 1, H/8, W/8, D]
        fusion_cost = self.aggregator(cost_volume)
        # [N, 1, H/8, W/8]
        disparity0 = self.computer(fusion_cost)

        height, width = left_image.shape[-2:]
        left_image_4x = F.interpolate(left_image, size=(height // 4, width // 4), mode='bilinear')
        left_image_2x = F.interpolate(left_image, size=(height // 2, width // 2), mode='bilinear')

        disparity1 = self.refiner1(left_image_4x, disparity0)  # 256*256
        disparity2 = self.refiner2(left_image_2x, disparity1)  # 512*512
        disparity = self.refiner3(left_image, disparity2)  # 1024*1024

        return disparity

    @torch.no_grad()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
