import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from .submodule import convbn_3d, disparityregression, feature_extraction


class PSMNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.min_disp = int(args.min_disp)
        self.max_disp = int(args.max_disp)
        self.disp_length = self.max_disp - self.min_disp
        self.feature_extraction = feature_extraction(args.in_channel)

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True)
        )

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):  # noqa: SIM114
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        b, c, h, w = refimg_fea.size()
        cost = Variable(torch.FloatTensor(b, c * 2, self.disp_length // 4, h, w).zero_()).cuda(left.device)

        for i in range(self.min_disp // 4, self.max_disp // 4):
            if i > 0:
                cost[:, :c, i - self.min_disp // 4, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, c:, i - self.min_disp // 4, :, i:] = targetimg_fea[:, :, :, :-i]
            elif i < 0:
                cost[:, :c, i - self.min_disp // 4, :, :i] = refimg_fea[:, :, :, :i]
                cost[:, c:, i - self.min_disp // 4, :, :i] = targetimg_fea[:, :, :, -i:]
            else:
                cost[:, :c, i - self.min_disp // 4, :, :] = refimg_fea
                cost[:, c:, i - self.min_disp // 4, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.interpolate(cost, [self.disp_length, left.size()[2], left.size()[3]], mode='trilinear')
        pred = disparityregression(self.min_disp, self.max_disp)(cost)

        return pred
