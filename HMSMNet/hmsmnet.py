import argparse

import torch
from torch import nn

from Module.computation import Estimation
from Module.cost import CostVolume
from .aggregation import FeatureFusion, Hourglass
from .feature import FeatureExtraction
from .refinement import Refinement


class HMSMNet(nn.Module):
    def __init__(self, args: argparse.Namespace, method='concat'):
        super().__init__()
        self.in_channel = args.in_channel
        self.channel = 16
        self.min_disp = int(args.min_disp)
        self.max_disp = int(args.max_disp)
        self.method = method

        self.feature_extraction = FeatureExtraction(self.in_channel, self.channel)
        # 论文中使用的构造方法为'difference'，但为了便于网络之间的对比，这里统一为'concat'
        self.cost0 = CostVolume(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, method=self.method)
        self.cost1 = CostVolume(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, method=self.method)
        self.cost2 = CostVolume(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, method=self.method)

        self.hourglass0 = Hourglass(self.channel * 2, self.channel)
        self.hourglass1 = Hourglass(self.channel * 2, self.channel)
        self.hourglass2 = Hourglass(self.channel * 2, self.channel)
        self.hourglass3 = Hourglass(self.channel, self.channel)
        self.hourglass4 = Hourglass(self.channel, self.channel)

        self.conv0 = nn.Conv3d(self.channel, 1, 3, 1, 'same')
        self.estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        self.conv1 = nn.Conv3d(self.channel, 1, 3, 1, 'same')
        self.estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.conv2 = nn.Conv3d(self.channel, 1, 3, 1, 'same')
        self.estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)

        self.fusion1 = FeatureFusion(self.channel)
        self.fusion2 = FeatureFusion(self.channel)
        self.refiner = Refinement(self.in_channel * 3 + 1, self.channel * 2)

        self._initialize_weights()

    def forward(self, left_image, right_image):
        """
        Args:
            left_image: 左图像
            right_image: 右图像

        Returns:
            list: [final_disp, disparity0, disparity1, disparity2]
        """
        l0, l1, l2 = self.feature_extraction(left_image)
        r0, r1, r2 = self.feature_extraction(right_image)

        cost_volume0 = self.cost0(l0, r0)
        cost_volume1 = self.cost1(l1, r1)
        cost_volume2 = self.cost2(l2, r2)

        agg_cost0 = self.hourglass0(cost_volume0)
        agg_cost1 = self.hourglass1(cost_volume1)
        agg_cost2 = self.hourglass2(cost_volume2)

        fusion_cost1 = self.fusion1(agg_cost2, agg_cost1)
        agg_fusion_cost1 = self.hourglass3(fusion_cost1)
        fusion_cost0 = self.fusion2(agg_fusion_cost1, agg_cost0)
        agg_fusion_cost0 = self.hourglass4(fusion_cost0)

        disparity2 = self.estimator2(self.conv2(agg_cost2))
        disparity1 = self.estimator1(self.conv1(agg_fusion_cost1))
        disparity0 = self.estimator0(self.conv0(agg_fusion_cost0))

        final_disp = self.refiner(left_image, disparity0)

        if self.training:
            return final_disp, disparity0, disparity1, disparity2
        else:
            return final_disp

    @torch.no_grad()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                nn.init.constant_(module.bias.data, 0)
