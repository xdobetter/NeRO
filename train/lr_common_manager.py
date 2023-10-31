import abc
import numpy as np


class LearningRateManager(abc.ABC):
    @staticmethod
    def set_lr_for_all(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network): # 构建优化器
        # may specify different lr for different parts
        # use group to set learning rate
        paras = network.parameters()
        return optimizer(paras, lr=1e-3)

    @abc.abstractmethod
    def __call__(self, optimizer, step, *args, **kwargs):
        pass


class WarmUpCosLR(LearningRateManager): # 根据当前步骤计算学习率，实现了预热和余弦退火的学习率调度策略
    default_cfg = {
        'end_warm': 5000,
        'end_iter': 300000,
        'lr': 5e-4,
    }

    def __init__(self, cfg):
        cfg = {**self.default_cfg, **cfg}
        self.warm_up_end = cfg['end_warm'] # 预热结束步骤
        self.learning_rate_alpha = 0.05 # 学习率衰减的最小值
        self.end_iter = cfg['end_iter'] # 总步骤
        self.learning_rate = cfg['lr'] # 基础学习率
    def __call__(self, optimizer, step, *args, **kwargs):
        if step < self.warm_up_end: # 学习率按照线性方式进行预热
            learning_factor = step / self.warm_up_end
        else: # 大于预热步数，学习率按照余弦方式进行衰减
            alpha = self.learning_rate_alpha
            progress = (step - self.warm_up_end) / (self.end_iter - self.warm_up_end) # 计算当前进度
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha # 计算学习率的衰减因子

        lr = self.learning_rate * learning_factor
        self.set_lr_for_all(optimizer, lr) # 应用于优化器的所有参数
        return lr


name2lr_manager = {
    'warm_up_cos': WarmUpCosLR,
}
