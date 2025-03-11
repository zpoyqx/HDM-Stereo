from torch import nn

from Module.conv import ResidualBlock, conv2d_bn


def make_blocks(channel, dilation, num):
    layers = []
    for _ in range(num):
        layers.append(ResidualBlock(channel, dilation=dilation))
    return nn.Sequential(*layers)


def avg_pool(in_channel, out_channel, kernel_size):
    pool = nn.AvgPool2d(kernel_size, kernel_size)
    conv = nn.Conv2d(in_channel, out_channel, 1)
    return nn.Sequential(pool, conv)


class FeatureExtraction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv0_1 = conv2d_bn(in_channel, out_channel, 5, 2, 2, 1, act_name='relu')
        self.conv0_2 = conv2d_bn(out_channel, out_channel * 2, 5, 2, 2, 1, act_name='relu')

        self.conv1_0 = make_blocks(out_channel * 2, 1, 4)
        self.conv1_1 = make_blocks(out_channel * 2, 2, 2)
        self.conv1_2 = make_blocks(out_channel * 2, 4, 2)
        self.conv1_3 = make_blocks(out_channel * 2, 1, 2)

        self.branch0 = avg_pool(out_channel * 2, out_channel, 1)
        self.branch1 = avg_pool(out_channel * 2, out_channel, 2)
        self.branch2 = avg_pool(out_channel * 2, out_channel, 4)

    def forward(self, inputs):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)

        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        return x0, x1, x2
