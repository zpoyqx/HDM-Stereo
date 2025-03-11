import torch
from torch.nn.functional import interpolate

from Tool.evaluation import myprint
from Tool.model import StereoModel


def translation(left, right, offset, extra_offset):
    pad = torch.zeros((*left.shape[:-1], abs(offset)), dtype=left.dtype, device=left.device)
    extra_pad = torch.zeros((*left.shape[:-1], extra_offset), dtype=left.dtype, device=left.device)
    if offset > 0:
        return torch.cat((extra_pad, left, pad), dim=-1), torch.cat((extra_pad, pad, right), dim=-1)
    elif offset < 0:
        return torch.cat((pad, left, extra_pad), dim=-1), torch.cat((right, pad, extra_pad), dim=-1)
    else:
        return left, right


def translation_estimate(model, left, right, offset, multiple=16):
    offset = int(offset)
    width = ((left.shape[-1] + abs(offset)) // multiple + 1) * multiple
    extra_offset = width - left.shape[-1] - abs(offset)
    output = model(*translation(left, right, offset, extra_offset))
    if offset > 0:
        return output[..., extra_offset:-offset] + offset
    elif offset < 0:
        return output[..., -offset:-extra_offset] + offset
    else:
        return output


def downsample(model, left, right, scale=2):
    # 下采样的分辨率最好和Random Crop保持一致
    # PSMNet的SPP模块有个64*64的平均池化层，在测试WHU数据集时会导致大小不够报错，需要改为1.5
    left_down = interpolate(left, scale_factor=1 / scale, mode='bilinear')
    right_dowm = interpolate(right, scale_factor=1 / scale, mode='bilinear')
    return model(left_down, right_dowm) * scale


def translate_optimize(model, left, right, isprint=True):
    if not isinstance(model, StereoModel):
        model = StereoModel(model, isprint=isprint)
    model.crop_costvolume(model.model_min_disp, model.model_max_disp)
    output_down = downsample(model, left, right)
    min_val, max_val = output_down.quantile(0.001).item(), output_down.quantile(0.999).item()
    model.crop_costvolume(min_val, max_val)
    output = model(left, right)
    offset = 0
    if max_val > model.model_max_disp:
        offset = min(min_val - model.model_min_disp, max_val)
    elif min_val < model.model_min_disp:
        offset = max(min_val, max_val - model.model_max_disp)
    if offset != 0:
        myprint(f'平移量: {offset}', isprint)
        model.crop_costvolume(min_val - offset, max_val - offset)
        output2 = translation_estimate(model, left, right, offset)
        mask = torch.abs(output2 - offset) < torch.abs(output2)
        output = torch.where(mask, output2, output)
    return output
