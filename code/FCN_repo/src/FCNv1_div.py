"""D-FCN：在 Tri-BCE 基础上加入分支去相关正则。

动机：实测 ECN 与 LCN 的 logit 相关系数为 0.93(iPinYou) / 0.97(KKBox)，
固定等权融合几乎无信息增量（alpha 扫描仅 +1e-4）。本变体通过显式惩罚
两分支的余弦相似度，迫使它们学习互补表征，从而让融合层真正起作用。

    L = TriBCE + lambda * cos(z_D, z_S)^2 * detach(TriBCE)
"""
import torch
from .FCNv1 import FCNv1


class FCNv1Div(FCNv1):
    def __init__(self, feature_map, div_lambda=0.05, **kwargs):
        super().__init__(feature_map, **kwargs)
        self.div_lambda = float(div_lambda)

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

        zd = torch.logit(y_d.clamp(1e-6, 1 - 1e-6))
        zs = torch.logit(y_s.clamp(1e-6, 1 - 1e-6))
        zd = zd - zd.mean()
        zs = zs - zs.mean()
        cos = (zd * zs).sum() / (zd.norm() * zs.norm() + 1e-8)
        return tri + self.div_lambda * cos.pow(2) * tri.detach()
