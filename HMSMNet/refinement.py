import torch
import torch.nn.functional as F
from torch import nn

from Module.conv import conv2d_bn


def concat(left_image, disp):
    # 对每个通道分别求Sobel梯度
    img_channels = torch.split(left_image, 1, 1)
    # kernel的第一个维度是out_channel，第二个维度是in_channel
    kernel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=left_image.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=left_image.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    gx = torch.cat([F.conv2d(img_channel, kernel_x, padding=1) for img_channel in img_channels], dim=1)
    gy = torch.cat([F.conv2d(img_channel, kernel_y, padding=1) for img_channel in img_channels], dim=1)
    return torch.cat([disp, left_image, gx, gy], dim=1)


class Refinement(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv1 = conv2d_bn(in_channel, channel, 3, 1, 'same', 1)
        self.conv2 = conv2d_bn(channel, channel, 3, 1, 'same', 1)
        self.conv3 = conv2d_bn(channel, channel, 3, 1, 'same', 2)
        self.conv4 = conv2d_bn(channel, channel, 3, 1, 'same', 3)
        self.conv5 = conv2d_bn(channel, channel, 3, 1, 'same', 1)
        self.conv6 = nn.Conv2d(channel, 1, 3, 1, 'same', 1)

    def forward(self, left_image, disp):
        scale_factor = left_image.shape[2] / disp.shape[2]
        disp = F.interpolate(disp, left_image.shape[-2:], mode='bilinear')
        disp = disp * scale_factor  # 联系视差值的定义思考

        inputs = concat(left_image, disp)
        delta = self.conv1(inputs)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        disp_final = disp + delta

        return disp_final
