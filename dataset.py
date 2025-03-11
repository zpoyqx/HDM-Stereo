import os
import os.path as osp

from torch.utils.data import Dataset

from Tool.data_reader import ImageReader


class StereoDataset(Dataset):
    def __init__(self, config, train=True):
        mode = 'train' if train else 'val'
        params = (config.aug, config.limit) if train else ()
        self.left_data_path = osp.join(config.data_path, mode, 'left')
        self.right_data_path = osp.join(config.data_path, mode, 'right')
        self.disp_data_path = osp.join(config.data_path, mode, 'disp')
        assert osp.exists(self.left_data_path) and osp.exists(self.right_data_path) and osp.exists(self.disp_data_path)

        self.files = os.listdir(self.left_data_path)
        self.data_reader = ImageReader(config.dataset, config.model_config.in_channel, train, *params)

    def __getitem__(self, index):
        left_file = osp.join(self.left_data_path, self.files[index])
        right_file = osp.join(self.right_data_path, self.files[index])
        disp_file = osp.join(self.disp_data_path, self.files[index])
        return self.data_reader.read(left_file, right_file, disp_file)

    def __len__(self):
        return len(self.files)
