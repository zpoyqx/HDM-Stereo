import argparse
import os.path as osp
from Tool.model import StereoModel
from eval import Evaluation


def batch_eval(config):
    model_dict = {
        'stereo': ['stereo-direct.pth', 'stereo-96.pth', 'stereo-192.pth'],
        'hmsm': ['hmsm-direct.pth', 'hmsm-96.pth', 'hmsm-192.pth'],
    }
    models = model_dict[config.model]

    config.checkpoint_path = osp.join(config.checkpoints_path, models[0])
    evaluator = Evaluation(config)
    evaluator.val_large_disp(optimize=False)
    evaluator.model = StereoModel(osp.join(config.checkpoints_path, models[1]), isprint=False)
    evaluator.val_large_disp(optimize=False)
    evaluator.model = StereoModel(osp.join(config.checkpoints_path, models[2]), isprint=False)
    evaluator.val_large_disp(optimize=False)
    evaluator.model = StereoModel(osp.join(config.checkpoints_path, models[0]), isprint=False, iscrop=False)
    evaluator.val_large_disp(optimize=True)
    evaluator.model = StereoModel(osp.join(config.checkpoints_path, models[1]), isprint=False, iscrop=False)
    evaluator.val_large_disp(optimize=True)
    evaluator.model = StereoModel(osp.join(config.checkpoints_path, models[1]), isprint=False)
    evaluator.val_large_disp(optimize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data-path', type=str, default='data', help='测试集路径(包含val文件夹)')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='使用的数据集')
    parser.add_argument('-m', '--model', type=str, required=True, help='使用的模型')
    parser.add_argument('-c', '--checkpoints-path', type=str, default='checkpoints', help='权重路径')
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-t', '--threshold', type=int, default=3, help='错误像素的判断阈值')
    parser.add_argument('-n', '--num-workers', type=int, default=0)
    config = parser.parse_args()

    batch_eval(config)