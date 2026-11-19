# -*- coding: utf-8 -*-
"""改进版 FCNv2 的代码包（本项目自研，与 `code/FCN/src` 原仓库并存、互不污染）。

用法：
    # 用改进版（默认开关 = 原版行为）
    from src.fcn_v2_soft import FCNv2Soft

    # 取原仓库的类做对照（只读加载，不改原仓库）
    from src._orig_loader import orig

注意包名：本目录在 `code/` 之下，所以当 `code/` 进入 sys.path 时，本包的名字是 `src`。
原仓库在 `code/FCN/src`，只有把 `code/FCN` 放进 sys.path 时它才叫 `src`。
两者靠 **sys.path 不同** 区分，绝不在同一进程里同时以 `src` 之名存在。
本文件另外用 `fcn_orig` 这个独立名字把原仓库加载进来（见 _orig_loader），
这样"原版 vs 改进版"可以在一个进程里逐位对比，而不必启动两次解释器。
"""

from __future__ import absolute_import

from .fcn_v2_soft import (
    FCNv2Soft,
    FCNv2,
    MultiHeadFeatureEmbedding,
    Exponential2LinearCrossNetwork,
    Linear2ExponentialCrossNetwork,
    TRI_BCE_MODES,
    resolve_projection_dim,
    resize_last_dim,
    cross_net_param_count,
)

__all__ = [
    "FCNv2Soft",
    "FCNv2",
    "MultiHeadFeatureEmbedding",
    "Exponential2LinearCrossNetwork",
    "Linear2ExponentialCrossNetwork",
    "TRI_BCE_MODES",
    "resolve_projection_dim",
    "resize_last_dim",
    "cross_net_param_count",
]
