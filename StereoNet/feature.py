import torch.nn as nn

from Module.conv import ResidualBlock, conv2d_bn


class FeatureExtraction(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv5x5_1 = conv2d_bn(in_channel, out_channel, 5, 2, 2, act_name='relu')
        self.conv5x5_2 = conv2d_bn(out_channel, out_channel, 5, 2, 2, act_name='relu')
        self.conv5x5_3 = conv2d_bn(out_channel, out_channel, 5, 2, 2, act_name='relu')

        self.res_1 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')
        self.res_2 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')
        self.res_3 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')
        self.res_4 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')
        self.res_5 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')
        self.res_6 = ResidualBlock(out_channel, dilation=1, act_name='leaky_relu')

        self.conv3x3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=True)

    def forward(self, inputs):
        outputs = self.conv5x5_1(inputs)
        outputs = self.conv5x5_2(outputs)
        outputs = self.conv5x5_3(outputs)

        outputs = self.res_1(outputs)
        outputs = self.res_2(outputs)
        outputs = self.res_3(outputs)
        outputs = self.res_4(outputs)
        outputs = self.res_5(outputs)
        outputs = self.res_6(outputs)

        outputs = self.conv3x3(outputs)
        return outputs
