import torch
from torch import nn
from torch.nn import functional as F


class CostVolume(nn.Module):
    def __init__(self, min_disp, max_disp, method, groups=8):
        super().__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.method = method
        self.groups = groups

    def compute(self, left_feature, right_feature):
        if self.method == 'difference':
            return left_feature - right_feature
        elif self.method == 'concat':
            return torch.cat((left_feature, right_feature), dim=1)
        elif self.method == 'correlation':
            B, C, H, W = left_feature.shape
            assert C % self.groups == 0
            channels_per_group = C // self.groups
            return (left_feature * right_feature).view([B, self.groups, channels_per_group, H, W]).mean(dim=2)
        elif self.method == 'mixed':
            ave_feature = (left_feature + right_feature) / 2
            ave_feature2 = (left_feature.pow(2) + right_feature.pow(2)) / 2
            variance = ave_feature2 - ave_feature.pow(2)
            return torch.cat((left_feature, right_feature, variance), dim=1)
        else:
            raise NotImplementedError

    def forward(self, left_feature, right_feature):
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(
                    F.pad(
                        input=self.compute(left_feature[:, :, :, :i], right_feature[:, :, :, -i:]),
                        pad=[0, -i, 0, 0, 0, 0],
                        mode='constant',
                    )
                )
            elif i > 0:
                cost_volume.append(
                    F.pad(
                        input=self.compute(left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                        pad=[i, 0, 0, 0, 0, 0],
                        mode='constant',
                    )
                )
            else:
                cost_volume.append(self.compute(left_feature, right_feature))
        cost_volume = torch.stack(cost_volume, -1)  # [B, C, H, W, D]
        return cost_volume
