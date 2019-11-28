"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch


class ShareAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.9), eps=1e-8, weight_decay=0):
        """
        :param params: 待优化参数的iterable或者参数组的字典
        :param lr: 学习率（默认：1e-3）
        :param betas: 计算梯度以及梯度平方的运行平均值的系数 （默认：0.9，0.999）
        :param eps: 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
        :param weight_decay: 权重衰减（L2惩罚）（默认: 0）
        """
        super(ShareAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:  # param_group:参数组(param), 学习率和各类学习参数
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
