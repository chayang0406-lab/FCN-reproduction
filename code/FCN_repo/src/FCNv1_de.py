"""DE-FCN：双分支独立嵌入（Dual-Embedding FCN）。

动机：FCN 两分支共享同一张 embedding 表、输入完全相同，导致输出高度相关
（实测 0.93 / 0.97），融合层无信息可加；而用损失惩罚强行去相关（D-FCN）
会牺牲分支精度（KKBox 上 LCN 损失上升 9%）。

本变体从源头断开共享：ECN 与 LCN 各用一张独立 embedding 表。去相关由结构
自然产生，不引入额外损失项。参数量增加一张 embedding 表。
"""
import torch
from .FCNv1 import FCNv1, MultiHeadFeatureEmbedding


class FCNv1DE(FCNv1):
    def __init__(self, feature_map, embedding_dim=16, num_heads=1, **kwargs):
        super().__init__(feature_map, embedding_dim=embedding_dim,
                         num_heads=num_heads, **kwargs)
        # 移除共享 embedding，换成两张独立的（保留 "embedding_layer" 前缀，
        # 使 BaseModel 的 embedding_regularizer 仍作用到它们）
        del self.embedding_layer
        self.embedding_layer_ecn = MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads)
        self.embedding_layer_lcn = MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads)
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        ecn_logit = self.ECN(self.embedding_layer_ecn(X)).mean(dim=1)
        lcn_logit = self.LCN(self.embedding_layer_lcn(X)).mean(dim=1)
        logit = (ecn_logit + lcn_logit) * 0.5
        y_pred = self.output_activation(logit)
        return {"y_pred": y_pred,
                "y_d": self.output_activation(ecn_logit),
                "y_s": self.output_activation(lcn_logit)}
