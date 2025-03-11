import os
import os.path as osp
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import wandb
from rich.console import Console
from torch import nn, optim
from torch.nn import SyncBatchNorm
from torch.nn.functional import interpolate
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from HMSMNet import HMSMNet
from PSMNet import PSMNet
from StereoNet import StereoNet
from Tool.evaluation import evaluate


class Solver:
    def __init__(self, config, train_loader, val_loader):
        # dataset
        self.dataset = config.dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_loader.sampler
        self.val_sampler = val_loader.sampler if val_loader is not None else None

        # model
        self.model_name = config.model
        self.model_config = config.model_config
        self.min_disp = self.model_config.min_disp
        self.max_disp = self.model_config.max_disp
        self.model: nn.Module = nn.Module()
        self.optimizer: optim = None
        self.scheduler: optim.lr_scheduler = None
        self.lr = config.lr
        self.criterion = nn.SmoothL1Loss()

        # train and val setting
        self.max_epoch = config.max_epoch + 1
        self.val_step = config.val_step
        self.threshold = config.threshold
        self.verbose = config.verbose

        # distributed training setting
        self.dist_status = config.dist_status
        self.rank = dist.get_rank() if self.dist_status else 0
        self.world_size = dist.get_world_size() if self.dist_status else torch.cuda.device_count()
        self.is_main = self.rank == 0
        self.device = torch.device('cuda', self.rank)
        # build Model
        self.build_model()

        # wandb：数据记录
        self.wandb_status = config.wandb_status
        self.wandb_key = ''  # 对于单次使用，可以手动指定key，否则建议使用`wandb login`登录自己的账号
        self.run_name = datetime.now().strftime('%Y%m%d-%H%M')
        # wandb初始化参数
        # 仅当wandb_status为True且在主进程时，才会启用wandb
        self.run_config = {
            'project': 'Stereo-Matching',
            'notes': config.note,
            'mode': 'disabled'
            if not self.wandb_status or not self.is_main
            else ('offline' if config.offline else 'online'),
            'tags': [self.model_name, self.dataset],
            'config': {
                'in_channel': self.model_config.in_channel,
                'seed': config.seed if config.test else -1,
                'batch_size': config.batch_size,
                'val_metric': config.val_metric,
            },
            'save_code': True,
        }
        if config.restore_status is not None and config.id is not None:
            extra_config = {'id': config.id, 'resume': 'allow'}  # 仅当从wandb恢复权重时才设置resume和id
        else:
            extra_config = {'name': self.run_name}  # name参数会覆盖原有的name参数，因此从wandb恢复权重时不指定
        self.run_config.update(extra_config)
        # wandb初始化
        self.init_wandb()

        # save and restore
        self.save_status = config.save_status
        self.restore_status = config.restore_status
        # 如果启用wandb，则将权重存放在wandb.run.dir下，否则存放在默认路径
        if self.wandb_status:
            # 默认的模型名称为：模型-数据集-run_id.pth
            self.save_path = osp.join(wandb.run.dir, f'{self.model_name}-{self.dataset}-{wandb.run.id}.pth')
        else:
            self.save_path = osp.join(config.default_save_dir, f'{self.model_name}-{self.dataset}-{self.run_name}.pth')
        self.restore_path = config.restore_path
        self.val_metric = config.val_metric
        self.val_best = float('inf')
        self.checkponit_best = None

        # 显示
        self.console = Console()
        self.locals = config.locals

    def build_model(self):
        if self.model_name == 'hmsm':
            self.model = HMSMNet(self.model_config)
        elif self.model_name == 'stereo':
            self.model = StereoNet(self.model_config)
        elif self.model_name == 'psm':
            self.model = PSMNet(self.model_config)
        else:
            raise NotImplementedError(f'网络({self.model_name})未实现')
        self.model.cuda(self.device)

        if self.dist_status:
            # 将所有BatchNorm层转化为SyncBatchNorm层，保证多卡训练时BatchNorm的统计量一致
            # 它在所有GPU之间同步BatchNorm的统计量，使得每个GPU都能看到完整的批次数据
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank])
        else:
            self.model = DataParallel(self.model, device_ids=range(self.world_size))
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, amsgrad=True)
        # 当你修改调度器时，下面的step函数也要跟着修改
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        # self.scheduler = CosineAnnealingLR(self.optimizer, self.max_epoch - 1, eta_min=self.lr / 100)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=1, cooldown=2, min_lr=self.lr / 256)

    def init_wandb(self):
        if len(self.wandb_key) != 0:
            wandb.login(key=self.wandb_key)
        wandb.init(**self.run_config)
        wandb.define_metric('step')
        wandb.define_metric('loss', step_metric='step')
        wandb.define_metric('avgerr', step_metric='step')
        wandb.define_metric(f'bad{self.threshold}', step_metric='step')
        wandb.define_metric('lr', step_metric='step')

    def restore(self):
        # 返回值为开始的epoch序号
        if self.restore_status is None:
            return 1
        elif self.restore_status == 'all':
            # 将模型加载进CPU，减少显存占用
            checkpoint = torch.load(self.restore_path, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 只在断点恢复时赋值self.val_best，迁移训练则不用
            self.val_best = checkpoint.get('val', float('inf'))
            return checkpoint['epoch'] + 1
        elif self.restore_status == 'weight_only':
            checkpoint = torch.load(self.restore_path, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return 1
        else:
            raise ValueError('不是支持的加载策略')

    def save(self, epoch, val_data):
        # 只在主卡保存模型
        # 至少训练10个epoch后，才会保存模型
        if self.save_status and self.is_main and val_data < self.val_best and epoch >= 10:
            self.val_best = val_data
            # 注意：由于这里保存的是一个字典，因此加载权重时要指定`weights_only=False`
            self.checkponit_best = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'dataset': self.dataset,
                'model_config': self.model_config,
                'val_metric': self.val_metric,
                'val': val_data,
            }

    def to_cuda(self, tensors):
        # non_blocking=True和Dataloader的pin_memory=True搭配使用，可以加快CPU到GPU的数据传输
        if isinstance(tensors, (list, tuple)):
            return [tensor.cuda(self.device, non_blocking=True) for tensor in tensors]
        else:
            return tensors.cuda(self.device, non_blocking=True)

    def reduce_data(self, data, train):
        # 将各卡计算结果求和归约到主卡中
        if train:
            avg_data = torch.tensor(data / len(self.train_loader), dtype=torch.float32, device=self.device)
        else:
            avg_data = torch.tensor(data / len(self.val_loader), dtype=torch.float32, device=self.device)
        if self.dist_status:
            dist.reduce(avg_data, 0)
            avg_data = avg_data / self.world_size
        return avg_data.item()

    def compute_loss(self, outputs, target):
        if self.model_name == 'hmsm':
            weights = [0.6, 1, 0.7, 0.5]
            loss_sum = torch.tensor(0, dtype=torch.float32)
            for i, output in enumerate(outputs):
                # 注意视差的定义，上下采样都会导致视差值的scale发生变化
                scale_factor = output.shape[-1] / target.shape[-1]
                disp = target if i == 0 else interpolate(target, output.shape[-2:], mode='nearest-exact')
                mask = torch.ne(disp, -999.0).logical_and(disp >= self.min_disp).logical_and(disp <= self.max_disp)
                disp *= scale_factor
                loss = self.criterion(output * mask, disp * mask)
                loss_sum = loss_sum + loss * weights[i]
            return loss_sum
        elif self.model_name == 'psm':
            mask = torch.ne(target, -999.0).logical_and(target >= self.min_disp).logical_and(target <= self.max_disp)
            loss1 = self.criterion(outputs[0] * mask, target * mask)
            loss2 = self.criterion(outputs[1] * mask, target * mask)
            loss3 = self.criterion(outputs[2] * mask, target * mask)
            return 0.5 * loss1 + 0.7 * loss2 + loss3
        else:
            mask = torch.ne(target, -999.0).logical_and(target >= self.min_disp).logical_and(target <= self.max_disp)
            return self.criterion(outputs * mask, target * mask)

    @torch.no_grad()
    def val(self):
        self.model.eval()
        avgerr, bad = 0, 0
        if self.is_main and self.verbose:
            iter_loader = tqdm(self.val_loader, desc='val', file=sys.stdout, ncols=80)
        else:
            iter_loader = self.val_loader
        for inputs, target in iter_loader:
            inputs = self.to_cuda(inputs)
            # 默认网络输出的第一个值为原始分辨率视差图
            target = self.to_cuda(target)
            output = self.model(*inputs)
            errors = evaluate(output, target, self.threshold)
            avgerr += errors[0]
            bad += errors[1]
        self.model.train()
        return avgerr, bad

    def train(self):
        # 若wandb没有正常关闭，请在命令行执行：ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
        try:
            self._train()
            # 上传所有工程代码
            if self.wandb_status:
                path = os.getcwd()
                wandb.save(osp.join(path, '*.py'), base_path=osp.dirname(path))
                exclude_dirs = ['__pycache__', 'checkpoints', 'wandb']
                for file in os.listdir():
                    if osp.isdir(file) and (not file.startswith('.')) and file not in exclude_dirs:
                        wandb.save(osp.join(path, f'{file}/*'), base_path=osp.dirname(path))
        except Exception:
            self.console.print_exception(show_locals=self.locals)
        finally:
            # 保存权重
            if self.is_main and self.save_status and self.checkponit_best is not None:
                torch.save(self.checkponit_best, self.save_path)
                wandb.save(self.save_path, base_path=osp.dirname(self.save_path))
            # 结束wandb和分布式训练进程
            wandb.finish()
            if self.dist_status:
                dist.destroy_process_group()

    def _train(self):
        start_epoch = self.restore()
        for epoch in range(start_epoch, self.max_epoch):
            # 设置DistributedSampler的随机数种子，确保每个epoch都会重新随机采样
            if self.dist_status:
                self.train_sampler.set_epoch(epoch)
                if self.val_sampler is not None:
                    self.val_sampler.set_epoch(epoch)
            loss_sum = 0
            params = {'step': epoch}

            self.model.train()
            if self.is_main and self.verbose:
                iter_loader = tqdm(self.train_loader, desc=f'[{epoch}]train', file=sys.stdout, ncols=80)
            else:
                iter_loader = self.train_loader
            for inputs, target in iter_loader:
                self.optimizer.zero_grad()
                inputs = self.to_cuda(inputs)
                target = self.to_cuda(target)
                outputs = self.model(*inputs)
                loss = self.compute_loss(outputs, target)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()

            params['loss'] = self.reduce_data(loss_sum, True)
            if self.val_step != 0 and epoch % self.val_step == 0:
                avgerr, bad = self.val()
                params['avgerr'] = self.reduce_data(avgerr, False)
                params[f'bad{self.threshold}'] = self.reduce_data(bad, False)
                self.save(epoch, params[self.val_metric])
            
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                params['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.scheduler.step(params[self.val_metric])
            else:
                params['lr'] = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
            
            if self.is_main:
                wandb.log(params)
                self.console.print(params)

        if self.val_step == 0 or self.val_step >= self.max_epoch:
            self.checkponit_best = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model_config,
            }
