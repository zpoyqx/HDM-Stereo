import torch
import torch.nn.functional as F
from torch import nn


class Estimation(nn.Module):
    def __init__(self, min_disp, max_disp):
        super().__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs):
        # input_shape: [N, 1, H, W, D]
        length = self.max_disp - self.min_disp
        assert inputs.shape[-1] == length
        probabilities = F.softmax(-1.0 * inputs, dim=-1)
        candidates = torch.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, length, device=probabilities.device)
        disparities = torch.sum(candidates * probabilities, dim=-1)

        return disparities  # [N, 1, H, W]
