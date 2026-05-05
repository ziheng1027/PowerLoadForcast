"""
训练基类：通用训练流程框架。

子类通过覆写 forward() 实现模型特有逻辑。
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from tool.early_stopping import EarlyStopping


class TrainerBase:
    """训练基类，提供标准训练流程。"""

    def __init__():
        pass

    def _get_optimizer(self):
        """获取优化器。"""
        pass

    def _get_scheduler(self):
        """获取调度器。"""
        pass

    def _update_scheduler(self):
        """更新调度器。"""
        pass

    def train_batch(self):
        """训练一个 batch。"""
        pass

    def train_epoch(self):
        """训练一个 epoch。"""
        pass

    def train(self):
        """训练模型。"""
        pass

    def evaluate_batch(self, mode="valid"):
        """评估一个 batch。"""
        pass

    def valid(self):
        """在验证集上评估模型。"""
        pass

    def test(self):
        """在测试集上评估模型。"""
        pass