import argparse

import torch.nn as nn
import torch.nn.functional as F

from Module.computation import Estimation
from Module.cost import CostVolume
from .submodule import convbn_3d, feature_extraction


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super().__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, 3, 1, 1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(inplanes * 2),
        )  # +conv2
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(inplanes),
        )  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        pre = F.relu(pre + postsqu, inplace=True) if postsqu is not None else F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, args: argparse.Namespace, method='concat'):
        super().__init__()
        self.min_disp = int(args.min_disp)
        self.max_disp = int(args.max_disp)
        self.disp_length = self.max_disp - self.min_disp
        self.method = method

        self.feature_extraction = feature_extraction(args.in_channel)
        self.cost = CostVolume(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, method=self.method)
        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.computer = Estimation(min_disp=self.min_disp, max_disp=self.max_disp)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        cost = self.cost(refimg_fea, targetimg_fea)
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = F.interpolate(cost3, [*left.shape[2:4], self.disp_length], mode='trilinear')
        pred3 = self.computer(cost3)

        if self.training:
            cost1 = F.interpolate(cost1, [*left.shape[2:4], self.disp_length], mode='trilinear')
            cost2 = F.interpolate(cost2, [*left.shape[2:4], self.disp_length], mode='trilinear')
            pred1 = self.computer(cost1)
            pred2 = self.computer(cost2)
            return pred1, pred2, pred3
        else:
            return pred3
