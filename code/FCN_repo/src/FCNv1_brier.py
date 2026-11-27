"""Brier-FCN：在 Tri-BCE 之外加入 Brier 分数项，用于改善概率校准。

动机：BCE 只关心对数似然，对"整体概率偏移"惩罚极弱；实测 FCN 在 KKBox 上
ECE 达 2.73%，而调整融合权重最多只能降 7.7%。Brier 是 proper scoring rule，
直接惩罚 (p - y)^2。

    L = TriBCE + lambda * Brier(y_pred, y)
"""
import torch
from .FCNv1 import FCNv1


class FCNv1Brier(FCNv1):
    def __init__(self, feature_map, brier_lambda=1.0, **kwargs):
        super().__init__(feature_map, **kwargs)
        self.brier_lambda = float(brier_lambda)

    def add_loss(self, inputs):
        rd = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred, y_d, y_s = rd["y_pred"], rd["y_d"], rd["y_s"]

        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss_d = self.loss_fn(y_d, y_true, reduction='mean')
        loss_s = self.loss_fn(y_s, y_true, reduction='mean')
        wd = (loss_d - loss).clamp(min=0).detach()
        ws = (loss_s - loss).clamp(min=0).detach()
        tri = loss + loss_d * wd + loss_s * ws

        brier = torch.mean((y_pred.reshape(-1) - y_true.reshape(-1)) ** 2)
        return tri + self.brier_lambda * brier
