import os
import random
from pathlib import Path

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.renderer import name2renderer
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn


class Trainer:
    default_cfg = { #默认配置;它是一个类变量，所有实例共享同一个default_cfg
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 2.0e-4,
            "lr_step": 100000,
            "lr_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "novel_view_interval": 10000,
        "worker_num": 8, # load data 线程数
        'random_seed': 0, # default:6033
    }

    def _init_dataset(self): # 设置训练集和验证集
        self.train_set = name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True) # 创建train_set
        self.train_set = DataLoader(self.train_set, 1, True, num_workers=self.cfg['worker_num'],
                                    collate_fn=dummy_collate_fn) # 加载train_set;DataLoader返回一个可迭代的对象
        print(f'train set len {len(self.train_set)}') # 为什么这里先输出train set len，再输出val set len？
        self.val_set_list, self.val_set_names = [], []
        dataset_dir = self.cfg['dataset_dir']
        for val_set_cfg in self.cfg['val_set_list']: #遍历每种验证集，验证集这里看起来是可以有很多种？
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False, dataset_dir=dataset_dir) # 创建val_set
            val_set = DataLoader(val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn) # 加载val_set;
            self.val_set_list.append(val_set) # 加入一笔验证集
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self): # 设置网络
        self.network = name2renderer[self.cfg['network']](self.cfg).cuda() # 初始化网络架构的时候，将同时构造训练数据集

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam': # 设置优化器
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(self.optimizer, self.network) # 构建优化器

    def __init__(self, cfg): # 设置相关参数，如模型保存路径，随机种子等
        self.cfg = {**self.default_cfg, **cfg} # 合并配置，第1个配置是默认配置，第2个cfg来自配置文件yaml
        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir): Path(self.model_dir).mkdir(exist_ok=True, parents=True) # 创建该路径的目录，exist_ok表示如果目录存在，就不会抛出异常，parents=True参数表示会创建所有必要的父目录
        self.pth_fn = os.path.join(self.model_dir, 'model.pth') # 模型保存路径
        self.best_pth_fn = os.path.join(self.model_dir, 'model_best.pth') # 最佳模型保存路径

    def run(self): # train主函数，涵盖了整体的训练过程
        self._init_dataset()  # 初始化数据集，这里并没有构造训练数据集，只构造了题目所说的val数据集
        self._init_network() # 初始化网络，损失函数，metrics，val_evaluator，lr_manager
        self._init_logger() # 初始化日志

        best_para, start_step = self._load_model()
        train_iter = iter(self.train_set) # 生成迭代器 ; 在 Python 中，迭代器是一个实现了迭代协议的对象。它可以通过 iter() 函数来创建，该函数接受一个可迭代对象作为参数，并返回一个对应的迭代器对象。迭代器对象可以使用 next() 函数来逐个访问可迭代对象的元素，直到遍历完成

        pbar = tqdm(total=self.cfg['total_step'], bar_format='{r_bar}') # 包含 total 参数指定的总步骤数。进度条的格式由 bar_format 参数指定，其中 {r_bar} 表示进度条本身，即进度条的填充部分
        pbar.update(start_step)

        for step in range(start_step, self.cfg['total_step']): # 这样可以实现从中间处开始训练
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()  # 调用训练模式；这个train_network和network是同一个东西
            self.network.train() # 这里是不是重复了啊?这两个不是一个东西吗
            lr = self.lr_manager(self.optimizer, step) # 计算当前学习率

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            # if (step + 1) % self.cfg['novel_view_interval'] == 0:
            #     render_data = train_data.copy()
            #     render_data["render"] = True

            #     self.train_network(render_data)

            log_info = {}
            outputs = self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v

            loss = 0
            for k, v in log_info.items():
                if k.startswith('loss'):
                    print(f"[I] {k} v.requires_grad: ", v.requires_grad) # False
                    print("[I] v: ", torch.mean(v).item())
                    loss = loss + torch.mean(v) # 计算平均值
                    
            # loss.requires_grad_(True) # error method
            print("[I] loss: ", loss.item())
            print("[I] loss.requires_grad: ", loss.requires_grad) # False
            loss.backward()
            self.optimizer.step()
            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')

            if (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the final val set to select model!
                    val_para = val_para_cur

                if val_para > best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para = val_para
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                save_fn = None
                self._save_model(step + 1, best_para, save_fn=save_fn)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()

    def _load_model(self):
        best_para, start_step = 0, 0
        if os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_fn)

    def _init_logger(self): # 控制log的输出
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
