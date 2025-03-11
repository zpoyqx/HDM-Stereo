import torch.nn as nn

default_act_name = 'leaky_relu'
default_negative_slope = 0.2

def set_default_activation(act_name, negative_slope=None):
    global default_act_name, default_negative_slope
    default_act_name = act_name
    if negative_slope is not None:
        default_negative_slope = negative_slope


def get_activation(act_name=default_act_name):
    if act_name is None:
        return None
    elif act_name == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=default_negative_slope, inplace=True)
    elif act_name == 'swish':
        return nn.Hardswish(inplace=True)
    elif act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'selu':
        return nn.SELU(inplace=True)
    elif act_name == 'mish':
        return nn.Mish(inplace=True)
    else:
        raise ValueError(f'{act_name} is not supported')


def conv2d_bn(in_channel, out_channel, kernel_size, stride, padding, dilation=1, groups=1, act_name=default_act_name):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=False)
    bn = nn.BatchNorm2d(out_channel)
    act_layer = get_activation(act_name)
    if act_layer:
        return nn.Sequential(conv, bn, act_layer)
    else:
        return nn.Sequential(conv, bn)


def trans_conv2d_bn(
        in_channel, out_channel, kernel_size, stride, padding, output_padding=0, dilation=1, act_name=default_act_name
):
    conv = nn.ConvTranspose2d(
        in_channel, out_channel, kernel_size, stride, padding, output_padding, dilation=dilation, bias=False
    )
    bn = nn.BatchNorm2d(out_channel)
    act_layer = get_activation(act_name)
    if act_layer:
        return nn.Sequential(conv, bn, act_layer)
    else:
        return nn.Sequential(conv, bn)


def conv3d_bn(in_channel, out_channel, kernel_size, stride, padding, dilation=1, groups=1, act_name=default_act_name):
    conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=False)
    bn = nn.BatchNorm3d(out_channel)
    act_layer = get_activation(act_name)
    if act_layer:
        return nn.Sequential(conv, bn, act_layer)
    else:
        return nn.Sequential(conv, bn)


def trans_conv3d_bn(
        in_channel, out_channel, kernel_size, stride, padding, output_padding=0, dilation=1, act_name=default_act_name
):
    conv = nn.ConvTranspose3d(
        in_channel, out_channel, kernel_size, stride, padding, output_padding, dilation=dilation, bias=False
    )
    bn = nn.BatchNorm3d(out_channel)
    act_layer = get_activation(act_name)
    if act_layer:
        return nn.Sequential(conv, bn, act_layer)
    else:
        return nn.Sequential(conv, bn)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, channel=None, dilation=1, act_name='relu'):
        super().__init__()
        channel = channel or in_channel
        self.conv1 = conv2d_bn(in_channel, channel, 3, 1, 'same', dilation, act_name=act_name)
        self.conv2 = conv2d_bn(channel, channel, 3, 1, 'same', dilation, act_name=None)
        self.activation = get_activation()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.activation(x)
        return x
