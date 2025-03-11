from torch import nn

from Module.conv import conv3d_bn


class FeatureFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv3d_1 = conv3d_bn(in_channel, out_channel, 3, 1, 'same')
        self.conv3d_2 = conv3d_bn(out_channel, out_channel, 3, 1, 'same')
        self.conv3d_3 = conv3d_bn(out_channel, out_channel, 3, 1, 'same')
        self.conv3d_4 = conv3d_bn(out_channel, out_channel, 3, 1, 'same')

        self.conv3d = nn.Conv3d(out_channel, 1, 3, 1, 'same', 1, bias=False)

    def forward(self, inputs):
        outputs = self.conv3d_1(inputs)
        outputs = self.conv3d_2(outputs)
        outputs = self.conv3d_3(outputs)
        outputs = self.conv3d_4(outputs)
        outputs = self.conv3d(outputs)

        return outputs
