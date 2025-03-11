import torch
import torch.nn.functional as F
from torch import nn

from Module.conv import ResidualBlock, conv2d_bn


class Refinement(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        # in_channels: [low disparity, resized left image]的通道数
        self.conv1 = conv2d_bn(in_channel, channel, 3, 1, 'same')
        self.res_1 = ResidualBlock(channel, dilation=1, act_name='leaky_relu')
        self.res_2 = ResidualBlock(channel, dilation=2, act_name='leaky_relu')
        self.res_3 = ResidualBlock(channel, dilation=4, act_name='leaky_relu')
        self.res_4 = ResidualBlock(channel, dilation=8, act_name='leaky_relu')
        self.res_5 = ResidualBlock(channel, dilation=1, act_name='leaky_relu')
        self.res_6 = ResidualBlock(channel, dilation=1, act_name='leaky_relu')
        self.conv2 = nn.Conv2d(channel, 1, 3, 1, 'same', 1)

    def forward(self, left_image, disp):
        # disparity: [N, 1, H, W]
        # inputs: [low disparity, resized left image], [(N, 1, H, W), (N, 1, H, W)]

        scale_factor = left_image.shape[2] / disp.shape[2]
        disp = F.interpolate(disp, size=left_image.shape[-2:], mode='bilinear', align_corners=True)
        disp = disp * scale_factor

        concat = torch.cat([left_image, disp], dim=1)
        delta = self.conv1(concat)
        delta = self.res_1(delta)
        delta = self.res_2(delta)
        delta = self.res_3(delta)
        delta = self.res_4(delta)
        delta = self.res_5(delta)
        delta = self.res_6(delta)
        delta = self.conv2(delta)

        disp_final = disp + delta

        return disp_final
