import torch.nn.functional as F
from torch import nn

from Module.conv import conv3d_bn


class Hourglass(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = conv3d_bn(in_channel, out_channel, 3, 1, 1)
        self.conv2 = conv3d_bn(out_channel, out_channel, 3, 1, 1)
        self.conv3 = conv3d_bn(out_channel, out_channel * 2, 3, 2, 1)
        self.conv4 = conv3d_bn(out_channel * 2, out_channel * 2, 3, 1, 1)
        self.conv5 = conv3d_bn(out_channel * 2, out_channel * 2, 3, 2, 1)
        self.conv6 = conv3d_bn(out_channel * 2, out_channel * 2, 3, 1, 1)
        self.conv7 = conv3d_bn(out_channel * 2, out_channel * 2, 3, 1, 1)
        self.conv8 = conv3d_bn(out_channel * 2, out_channel, 3, 1, 1)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x4 = F.interpolate(x3, x2.shape[-3:], mode='nearest-exact')
        x4 = self.conv7(x4)
        x4 = x4 + x2
        x5 = F.interpolate(x4, x1.shape[-3:], mode='nearest-exact')
        x5 = self.conv8(x5)
        x5 = x5 + x1

        return x5


class FeatureFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features=channel, out_features=channel, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=channel, out_features=channel, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        x1 = F.interpolate(low, high.size()[-3:], mode='nearest-exact')
        x2 = x1 + high
        shape = x2.size()
        v = self.avg_pool3d(x2)
        v = v.reshape(shape[0], -1)
        v = self.fc1(v)
        v = self.activation(v)
        v = self.fc2(v)
        v = self.sigmoid(v)
        v1 = 1.0 - v
        v = v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        v1 = v1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = v * x1 + v1 * high
        return x
