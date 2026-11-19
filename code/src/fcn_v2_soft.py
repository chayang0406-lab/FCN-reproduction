# -*- coding: utf-8 -*-
"""FCNv2 的两处小改进（Tri-BCE 软权重 + LCA 压缩率），一个文件、两个开关。

===============================================================================
这份文件是什么
===============================================================================
原仓库 `code/FCN/src/FCNv2.py` 的训练目标里有一处"硬截断"，我们把它换成平滑版本；
另外把 LCA 里写死的"维度砍半"参数化，用来画参数量-AUC 曲线。

**本文件不改动原仓库任何字节。** 原版在这里被"重新实现一遍"，并靠开关保证
`tri_bce_mode="hard"` + `lca_compress_ratio=0.5` 与原版**逐位相同**
（见 code/src/tests/test_bit_exact.py，用 torch.equal 而非 allclose 断言）。

为什么不用"继承 + 覆盖"的写法：
    要改的那行在 `_cross_layer` 内部，继承只能整段重写；而且若先建原版交叉网络
    再替换成新版，会多消耗一轮随机数，权重就对不上，"逐位一致"自动失效。
    所以这里把结构原样抄一遍，只让开关真正生效。

===============================================================================
改动一：Tri-BCE 的权重软化（主贡献）
===============================================================================
原版（FCNv2.add_loss，原文第 86-90 行）：
    weight = max(0, L_aux - L_main)
    loss   = L_main + L_d * weight_d + L_s * weight_s

用大白话说：FCNv2 有两个"单塔"（只看指数网络 E2L 的一路、只看线性网络 L2E 的一路），
外加一个把两路平均起来的"合体"预测。原版的做法是——
**哪个单塔比合体预测差，就额外罚它一下，罚的力度 = 差多少。**

问题出在那个 max(0, ·)：
    差值为正时力度线性增长，差值为负时力度**瞬间归零**。
    在"差值恰为 0"处这条曲线有个尖角（不可导）：梯度从 0 跳到约 1。
    训练上表现为损失曲线出现小毛刺；更麻烦的是"死区"——
    权重一旦归零，即使那个单塔后来变好了，梯度仍是 0，它再也拿不到额外监督。

软权重的思路：把"归零"改成"变小但不断"，让梯度处处存在。两种实现：

  (1) `softplus` —— w = alpha · (1/β) · log(1 + exp(β·Δ))，Δ = L_aux - L_main
      人话：Δ 很负时权重趋近 0 但**不等于** 0（留着一根很细的梯度线）；
            Δ 很正时权重约等于 Δ，与计算原版一致。
      **为什么这个形式特别值得讲**：β→∞ 时它收敛到 relu(Δ)，也就是原版的硬截断。
      换句话说，**原版是软权重的一个极端情形**，我们做的是"把硬度旋钮从无穷大调到有限"。
      这也给出退火的自然解释：先把 β 调小（很软），再逐步调大（回到硬），
      而不是硬生生从一个目标跳到另一个目标。
  (2) `ratio` —— w = alpha · Δ / (L_aux + L_main)
      人话：不看绝对差值，看"差多少相对于两个损失之和有多大"。好处是量纲无关
           （损失整体放大 10 倍时权重不变），坏处是**可能取负值**——
           单塔明显更好时反而给它加权，会让训练不稳，故提供
           `tri_bce_clamp_nonneg` 开关。
  (3) `smoothmax` —— w = alpha · Δ · sigmoid(β·Δ)，也就是"把 relu 换成软 relu 但去掉常数项"
      为什么必须有这一档（这是实测出来的，不是纸上推的）：
          softplus 展开是 softplus(βΔ)/β = relu(Δ) + log(2)/β + O(Δ²)。
          中间那个 log(2)/β 是个**与 Δ 无关的常数偏置**。
          Criteo 上损失约 0.436，Δ = L_aux - L_main 的典型量级只有 1e-4 ~ 1e-3。
          若取 β=1，则 log(2)/β = 0.693 ——比 Δ 本身大三个数量级，
          软权重会恒等于 0.69 左右，等于"不管单塔好坏都狠狠加权"，训练必然崩。
          所以 softplus 必须把 β 调到与 Δ 尺度匹配（β≈1e3~1e4），
          或者直接用本档：Δ·sigmoid(βΔ) 在 Δ→0 时 ≈ Δ/2 - βΔ²/4，**没有常数项**，
          量纲天然正确，β 只需 ~1 量级就能用。
      人话：softplus 像"把地面抬高了 log(2)/β 那么厚"，走在下坡时脚下也还踩着土；
           smoothmax 像"把地面压回 0"，只把拐角磨圆。

退火（可选）：w_k = w · [floor + (1 - floor)·exp(-λ·k)]，k = 已完成的 epoch 数。
    人话：改动初期用完整强度，越往后越接近不加权（或衰减到 floor）。
    理由是辅助项的主要价值在训练早期（帮两个单塔尽快成形），后期合体与单塔已差不多好，
    再使劲加权只会拖累收敛。
    实现上直接读 BaseModel 已有的 `self._epoch_index`，**不改框架任何代码**；
    取不到时（比如单测直接调 add_loss）按 k=0 处理，λ=0 时该分支对数值零影响。

===============================================================================
改动二：LCA 压缩率 r（支撑分析）
===============================================================================
原版每个交叉层固定：
    w[i] = nn.Linear(D, D // 2, bias=False)      # D = input_dim = 字段数 × embedding_dim
    H1 = w[i](x)                                 # 宽度 d = D/2
    H2 = γ ⊙ H1 + β                              # 宽度 d（或 LayerNorm）
    H  = concat([H1, H2])                        # 宽度 d + d = D，**正好**接得上残差
    x  = x_anchor ⊙ (H + b) + x                  # b 宽度 D，残差流宽度始终是 D

关键观察：那个 `//2` 不是随手写的——**只有砍半，拼起来的宽度才正好等于 D**。
所以把 r 参数化之后，宽度必须再调回 D，否则 `H + b` 直接形状报错。

我们的做法：**加一个"零参数"的宽度适配器**，把宽度 2rD 映射回 D：
    r = 1/2  → 2rD = D，恒等（**一个算子都不做**，这是逐位一致的关键）
    r < 1/2  → 沿最后一维平铺（tile）后截到 D：把压缩后的交互信号"广播"回 D 个坐标
    r > 1/2  → 零填充到整数倍再按组求和（fold）折回 D：把过完备信号聚合回 D 维
为什么坚持"零参数"：这样参数量严格正比于 r，**参数量-AUC 曲线才是干净的**，
    唯一在变的自变量就是"可学习投影的宽度"。适配器若带参数，r 与参数量之间就混进了
    第二个变量，曲线无法解释。

参数量解析式（代码可自检，见 tools/lca_param_formula.py）：
    每个交叉层 = D·d + 2d + D      （w 的权重 + γ/β + b），d = r·D
    交叉网络   = L·(D·d + 2d + D) + (D + 1)，L = exp_num_layers + lin_num_layers
    r=1 与 r=1/2 之比的极限是 2（D² 项翻倍），有限 D 下略小于 2
    （3D 那一项不翻倍），例如 D=624 时是 1.994。
    ⚠️ 会影响报告写法的事实（实测，见 test_B1）：在 Criteo_x4_10_h5 上，
    总参数的 88.1% 是 39 个特征的词表 embedding（14,571,952），交叉网络合计 1,960,610。
    所以 r 从 1/2 变到 1，**模型总参数只涨约 11.8%**，不是 2 倍。

    ===== r=1 走独立的「全秩」路径（不带 γ/β）=====
    这一点不是偏好，是对齐论文 Table 5 的必然结论。论文 Criteo 四个变体反解：
        ECN-full（4 层）：不带 γ/β = 4×390,000+625 = 1,560,625 → 1.56M  ← 论文写 1.56M ✓
                        带  γ/β = 4×391,248+625 = 1,565,617 → 1.57M     ← 论文写 1.56M ✗
        FCNp-full（9 层）：不带 γ/β = 9×390,000+625 = 3,510,625 → 3.51M ← 论文写 3.51M ✓
                         带  γ/β = 9×391,248+625 = 3,521,857 → 3.52M    ← 论文写 3.51M ✗
    按两位小数四舍五入，**两处都只与「不带 γ/β」吻合**，带 γ/β 会多出 0.01M 对不上。
    语义上也自洽：γ/β 是 LCA 为了"砍半后补偿表达能力"加的配套仿射
    （论文原话 "introduce a low-cost affine transformation to maintain consistency"），
    全秩时根本不存在维度不一致，自然不需要它。

    所以 r=1 的结构等同于 ECN_full.py 里的 ExponentialCrossNetwork：
        H = w(x)                # w: D→D，无 γ/β，也不做 concat
        x = x_anchor ⊙ (H + b) + x
    这带来一个**结构上的不连续**：r∈{1/4,1/2,3/4} 是 LCA 结构（带 γ/β + concat + 宽度适配），
    r=1 是全秩结构（无 γ/β、无 concat）。报告必须如实声明这个不连续，
    否则"参数量随 r 单调变化"会被质疑。

    ===== 改完之后的比例（不再"恰好 2.00"）=====
    5 层：r=1/2 = 980,305，r=1 = 1,950,625，比值 1.9898
    9 层：r=1/2 = 1,764,049，r=1 = 3,510,625，比值 1.9901
    原因：r=1 少了 γ/β 那一份，所以比值从 2.000 掉到约 1.99。
    这与论文自己的表述一致：ECN 0.78→1.56 是 2.00×，FCN𝑝 1.76→3.51 是 1.99×。
    **不要硬凑 2.00**；1.99 才是对的。

===============================================================================
配置项（全部有默认值，默认值 = 原版行为）
===============================================================================
    tri_bce_mode:          "hard" | "softplus" | "ratio" | "smoothmax" | "plain"  (默认 "hard")
    tri_bce_alpha:         soft 模式下的总强度 alpha                  (默认 1.0)
    tri_bce_beta:          softplus/smoothmax 的锐度 β                 (默认 1.0)
    tri_bce_eps:           ratio 模式分母的安全下界                    (默认 1e-8)
    tri_bce_clamp_nonneg:  ratio 模式是否把负权重截成 0                (默认 False)
    tri_bce_anneal_lambda: 退火速率 λ，0 表示不退火                    (默认 0.0)
    tri_bce_anneal_floor:  退火下界                                    (默认 0.0)
    tri_bce_log_weights:   是否记录每步平均权重（会强制同步，略慢）     (默认 False)
    lca_compress_ratio:    压缩率 r ∈ (0, 1]，0.5 = 原版               (默认 0.5)
"""

from __future__ import absolute_import

import math

import torch
from torch import nn
from torch.nn import functional as F

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding

__all__ = [
    "FCNv2Soft",
    "FCNv2",  # 同名别名：配置里写 model: FCNv2 就能指到本类（同接口，drop-in）
    "MultiHeadFeatureEmbedding",
    "Exponential2LinearCrossNetwork",
    "Linear2ExponentialCrossNetwork",
    "resolve_projection_dim",
    "resize_last_dim",
    "cross_net_param_count",
    "TRI_BCE_MODES",
]

# 五种目标函数形态
#   hard      : 原版 max(0, Δ) 硬截断（默认，逐位等价）
#   softplus  : alpha·softplus(βΔ)/β，β→∞ 时即原版；但含 log(2)/β 常数偏置
#   smoothmax : alpha·Δ·sigmoid(βΔ)，软 relu 但无常数偏置（推荐作为主改进）
#   ratio     : alpha·Δ/(L_aux + L_main)，量纲无关，可能为负
#   bsw       : alpha·tanh(beta·Δ/L_main)，**有界**，处处可导，量纲无关（本文方案B）
#   plain     : 去掉两个辅助项，退化成普通二元交叉熵（论文消融里的 w/o Tri-BCE）
TRI_BCE_MODES = ("hard", "softplus", "smoothmax", "ratio", "bsw", "plain")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def resolve_projection_dim(input_dim, ratio):
    """把压缩率 r 换算成投影宽度 d = r·D。

    为什么 r=0.5 要单独特判成 `input_dim // 2`：
        原版写的就是 Python 整除 `//2`。它对偶数 D 与 round(D/2) 相同，
        但对奇数 D 是向下取整，而 `round()` 遇到 .5 走"四舍六入五成双"，会差 1 维。
        为了"开关默认时逐位一致"这条硬验收，这里必须原样照抄整除。

    注意：原版对奇数 input_dim 其实**跑不起来**（cat 后宽度 D-1，加 b（宽度 D）会报形状错），
        所以真实配置里 D 必为偶数（Criteo：39×16=624）。我们的 tile 分支顺带让奇数 D 也能跑，
        属于超集，不影响偶数情形的一致性。
    """
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("lca_compress_ratio 必须落在 (0, 1]，收到 %r" % (ratio,))
    if abs(ratio - 0.5) < 1e-12:
        return input_dim // 2          # ← 与原仓库字面一致，保证默认配置逐位相同
    d = int(round(input_dim * ratio))
    return max(1, min(d, input_dim))   # 夹到 [1, D]：太小没意义，超过 D 也无意义


def resize_last_dim(x, dst_dim):
    """把张量最后一维调到 dst_dim，**零参数、零可学习量**。

    src < dst → 平铺（tile）后截断：把压缩后的信号周期性铺到 dst 个坐标
    src > dst → 零填充到整数倍后按组求和（fold）：把过完备信号聚合回 dst 维
    src = dst → 原样返回（恒等，**不执行任何算子**，这是逐位一致的关键）
    """
    src_dim = x.shape[-1]
    if src_dim == dst_dim:
        return x
    if src_dim < dst_dim:
        k = int(math.ceil(dst_dim / float(src_dim)))
        return torch.cat([x] * k, dim=-1)[..., :dst_dim]
    rem = src_dim % dst_dim
    if rem:
        x = F.pad(x, (0, dst_dim - rem))   # 补零到整数倍，避免截断丢信息
    k = x.shape[-1] // dst_dim
    return x.reshape(*x.shape[:-1], dst_dim, k).sum(dim=-1)


def cross_net_param_count(input_dim, exp_num_layers, lin_num_layers, ratio):
    """一条交叉网络的**解析**参数量，用来和实测值对账（也是 r 扫描的自检）。

    LCA 结构（r < 1，带 γ/β + concat + 宽度适配）：
        每层: D·d + 2d + D   （w 权重 + γ/β + b；b 宽度 D）
        末尾: D + 1           （fc 的权重 + 偏置）
    全秩结构（r == 1，不带 γ/β、不做 concat）：
        每层: D·D + D         （w 权重 + b）
        末尾: D + 1
    两者的分界在 r=1，理由见模块 docstring 里"r=1 走独立全秩路径"一节。
    """
    L = int(exp_num_layers) + int(lin_num_layers)
    if is_full_rank(ratio):
        return L * (input_dim * input_dim + input_dim) + (input_dim + 1)
    d = resolve_projection_dim(input_dim, ratio)
    return L * (input_dim * d + 2 * d + input_dim) + (input_dim + 1)


def is_full_rank(ratio):
    """r == 1 时走全秩路径（无 γ/β、无 concat、无宽度适配）。"""
    return abs(float(ratio) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 与原仓库逐字相同的多头部特征 embedding
# ---------------------------------------------------------------------------
class MultiHeadFeatureEmbedding(nn.Module):
    """多头部特征 embedding（与 FCNv2.py 完全一致，未做改动）。

    做的是：把每个字段的 embedding 向量切成 H 份（H=num_heads），再把"前半段字段"和
    "后半段字段"分别拼接，得到 B×H×(F·D/H) 的张量。
    人话：让每个头各自看到一份独立的特征子空间，避免所有头挤在一起看同样的东西。
    保留它的原因是交叉网络层假设输入是三维 (batch, head, dim)。
    """

    def __init__(self, feature_map, embedding_dim, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

    def forward(self, X):  # H = num_heads
        feature_emb = self.embedding_layer(X)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
            multihead_feature_emb, 2, dim=-1
        )  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = (
            multihead_feature_emb1.flatten(start_dim=2),
            multihead_feature_emb2.flatten(start_dim=2),
        )  # B × H × FD/2H
        multihead_feature_emb = torch.cat(
            [multihead_feature_emb1, multihead_feature_emb2], dim=-1
        )
        return multihead_feature_emb  # B × H × FD/H


# ---------------------------------------------------------------------------
# 交叉网络（把原版两段几乎逐字相同的代码抽成公共基类，行为不变）
# ---------------------------------------------------------------------------
class _ParametricCrossNetwork(nn.Module):
    """指数/线性交叉网络的公共骨架，唯一改动是投影宽度由 r 决定。

    数学含义（一层）：
        H1 = W x                      # W: D→d，可学习；这是"交互信号"
        H2 = γ ⊙ H1 + β               # 逐元素仿射，可学习；或换成 LayerNorm
        H  = Resize([H1; H2])         # 零参数，把 2d 调回 D
        x' = x_anchor ⊙ (H + b) + x   # b: D，可学习；两条支路的差别只在这个 x_anchor
        logit = fc(x_final)           # fc: D→1

    两条支路的关系（FCN 的核心设计，不讲清楚就看不懂改动的含义）：
        E2L（指数→线性）：前 exp 层用 x_anchor = x（自己乘自己，连乘 → 指数级高阶），
                          后 lin 层用 x_anchor = x1（乘回最初输入 → 退化成线性项）。
        L2E（线性→指数）：顺序反过来。
        人话：一条先学"高阶组合"再学"低阶修正"，另一条反过来；
              两条一起跑相当于同时试两种归纳偏置，最后取平均。
    """

    def __init__(self,
                 input_dim,
                 exp_num_layers=3,
                 lin_num_layers=3,
                 batch_norm=True,
                 layer_norm=False,
                 net_dropout=0.1,
                 num_heads=1,
                 compress_ratio=0.5):
        super(_ParametricCrossNetwork, self).__init__()
        self.exp_num_layers = exp_num_layers
        self.lin_num_layers = lin_num_layers
        self.input_dim = input_dim
        self.compress_ratio = float(compress_ratio)
        # r == 1 走全秩路径：只建 w(D→D) 与 b(D)，**不建 γ/β**（对齐论文 -full 口径）
        self.full_rank = is_full_rank(compress_ratio)
        # d = r·D：可学习投影的宽度。非全秩且 r=0.5 时 d = D//2，即原版 LCA。
        self.proj_dim = input_dim if self.full_rank else \
            resolve_projection_dim(input_dim, compress_ratio)
        # 拼接后宽度 2d，需要调回 D。记成普通 bool（不是 nn.Module）：
        # 不进 state_dict、不加参数、不耗随机数。r=1/2 时它是 False → 完全不走适配分支；
        # 全秩路径不做 concat，因此也是 False。
        self.need_resize = (not self.full_rank) and (2 * self.proj_dim != input_dim)

        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()

        for i in range(exp_num_layers + lin_num_layers):
            # —— 必须保持与原版完全相同的"建模块顺序"：nn.Linear 构造会消耗随机数，
            #    顺序一变初始化就变，"逐位一致"就无从谈起。
            self.w.append(nn.Linear(input_dim, self.proj_dim, bias=False))
            if not self.full_rank:
                # γ/β 只在 LCA（r<1）路径存在。全秩时维度本就一致，不需要这份补偿仿射。
                if layer_norm:
                    self.layer_norm.append(nn.LayerNorm(self.proj_dim))
                else:
                    self.gamma.append(nn.Parameter(torch.ones(self.proj_dim)))
                    self.beta.append(nn.Parameter(torch.zeros(self.proj_dim)))
            self.b.append(nn.Parameter(torch.empty((input_dim,))))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.fc = nn.Linear(input_dim, 1)

    def _cross_layer(self, x, x_anchor, i):
        H_1 = self.w[i](x)
        if self.full_rank:
            # —— 全秩路径：无 γ/β、无 concat、无宽度适配。
            #    与 ECN_full.py 的 ExponentialCrossNetwork 逐字一致。
            H = H_1
        else:
            if len(self.layer_norm) > i:
                H_2 = self.layer_norm[i](H_1)
            else:
                H_2 = self.gamma[i] * H_1 + self.beta[i]
            H = torch.cat([H_1, H_2], dim=-1)
            # 零参数宽度适配。r=1/2 时 need_resize=False，**这个分支根本不进**，
            # 数值路径与原版完全一致（这是逐位一致的第二个关键点）。
            if self.need_resize:
                H = resize_last_dim(H, self.input_dim)
        if len(self.batch_norm) > i:
            H = self.batch_norm[i](H)
        x = x_anchor * (H + self.b[i]) + x
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        return x

    def extra_repr(self):
        return "input_dim=%d, proj_dim=%d (r=%.4g, full_rank=%s, resize=%s)" % (
            self.input_dim, self.proj_dim, self.compress_ratio,
            self.full_rank, self.need_resize)


class Exponential2LinearCrossNetwork(_ParametricCrossNetwork):
    """指数→线性 支路：先用自己乘自己（连乘出高阶），再乘回最初输入（低阶修正）。"""

    def forward(self, x):
        x1 = x
        for i in range(self.exp_num_layers):
            x = self._cross_layer(x, x, i)      # x_anchor = 当前 x → 连乘 → 指数级
        for i in range(self.lin_num_layers):
            i = i + self.exp_num_layers
            x = self._cross_layer(x, x1, i)     # x_anchor = 最初输入 → 线性项
        logit = self.fc(x)
        return logit


class Linear2ExponentialCrossNetwork(_ParametricCrossNetwork):
    """线性→指数 支路：顺序与上一条相反，得到互补的归纳偏置。"""

    def forward(self, x):
        x1 = x
        for i in range(self.lin_num_layers):
            x = self._cross_layer(x, x1, i)
        for i in range(self.exp_num_layers):
            i = i + self.lin_num_layers
            x = self._cross_layer(x, x, i)
        logit = self.fc(x)
        return logit


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class FCNv2Soft(BaseModel):
    """FCNv2 + 两个可切换的小改进。默认配置下与原仓库 FCNv2 完全等价。

    接口与原版一致（构造参数名、forward 返回的字典键 y_pred/y_d/y_s、add_loss 签名都不变），
    因此可以把配置里的 `model: FCNv2` 直接指到本类做 drop-in 替换，
    A/B 对照里唯一的差别就是几个开关的取值。
    """

    def __init__(self,
                 feature_map,
                 model_id="FCNv2",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 exp_num_layers=4,
                 lin_num_layers=4,
                 net_dropout=0.1,
                 batch_norm=False,
                 layer_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 # ——— 改动二的开关 ———
                 lca_compress_ratio=0.5,
                 # ——— 改动一的开关 ———
                 tri_bce_mode="hard",
                 tri_bce_alpha=1.0,
                 tri_bce_beta=1.0,
                 tri_bce_eps=1e-8,
                 tri_bce_clamp_nonneg=False,
                 tri_bce_anneal_lambda=0.0,
                 tri_bce_anneal_floor=0.0,
                 tri_bce_log_weights=False,
                 # ——— BSW-Tri 专用：分母防零 + 用相对差而非绝对差 ———
                 tri_bce_bsw_relative=True,
                 **kwargs):
        super(FCNv2Soft, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)

        # ——— 开关校验：模式名拼错要立刻报错，不能静默退回原版 ———
        mode = str(tri_bce_mode).lower()
        if mode not in TRI_BCE_MODES:
            raise ValueError("tri_bce_mode 必须是 %s 之一，收到 %r"
                             % (list(TRI_BCE_MODES), tri_bce_mode))
        self.tri_bce_mode = mode
        self.tri_bce_alpha = float(tri_bce_alpha)
        self.tri_bce_beta = float(tri_bce_beta)
        self.tri_bce_eps = float(tri_bce_eps)
        self.tri_bce_clamp_nonneg = bool(tri_bce_clamp_nonneg)
        self.tri_bce_anneal_lambda = float(tri_bce_anneal_lambda)
        self.tri_bce_anneal_floor = float(tri_bce_anneal_floor)
        self.tri_bce_log_weights = bool(tri_bce_log_weights)
        self.tri_bce_bsw_relative = bool(tri_bce_bsw_relative)
        self.lca_compress_ratio = float(lca_compress_ratio)
        # 最近一步的平均权重，仅在 tri_bce_log_weights=True 时更新
        # （供分析脚本读取，不参与任何计算）
        self.last_tri_bce_weights = None

        self.embedding_layer = MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads)
        input_dim = feature_map.sum_emb_out_dim()
        self.E2LCN = Exponential2LinearCrossNetwork(
            input_dim=input_dim,
            exp_num_layers=exp_num_layers,
            lin_num_layers=lin_num_layers,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            num_heads=num_heads,
            compress_ratio=self.lca_compress_ratio)
        self.L2ECN = Linear2ExponentialCrossNetwork(
            input_dim=input_dim,
            exp_num_layers=exp_num_layers,
            lin_num_layers=lin_num_layers,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            num_heads=num_heads,
            compress_ratio=self.lca_compress_ratio)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    # ------------------------------------------------------------------
    # 前向：与原版逐字相同，未做任何改动
    # ------------------------------------------------------------------
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        E2L_logit = self.E2LCN(feature_emb).mean(dim=1)
        L2E_logit = self.L2ECN(feature_emb).mean(dim=1)
        logit = (E2L_logit + L2E_logit) * 0.5
        y_pred = self.output_activation(logit)
        return_dict = {"y_pred": y_pred,
                       "y_d": self.output_activation(E2L_logit),
                       "y_s": self.output_activation(L2E_logit)}
        return return_dict

    # ------------------------------------------------------------------
    # 目标函数：全部改动只在 add_loss 里
    # ------------------------------------------------------------------
    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        loss = self.loss_fn(y_pred, y_true, reduction='mean')

        # —— plain：去掉两个辅助项，只剩普通 BCE（论文消融里的 w/o Tri-BCE）——
        if self.tri_bce_mode == "plain":
            return loss

        loss_d = self.loss_fn(y_d, y_true, reduction='mean')
        loss_s = self.loss_fn(y_s, y_true, reduction='mean')

        if self.tri_bce_mode == "hard":
            # ——— 原版代码，一字不改 ———
            # 连 `torch.zeros(1)` 这个形状为 (1,) 的零张量也保留：
            # 它会被广播，行为等价，但保留原写法可以让"逐位一致"更容易被审计。
            weight_d = loss_d - loss
            weight_s = loss_s - loss
            weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
            weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
            loss = loss + loss_d * weight_d + loss_s * weight_s
            if self.tri_bce_log_weights:
                self.last_tri_bce_weights = (float(weight_d.detach().mean()),
                                              float(weight_s.detach().mean()))
            return loss

        # ——— 以下是软化版本 ———
        weight_d = self._soft_weight(loss_d - loss, loss_d, loss)
        weight_s = self._soft_weight(loss_s - loss, loss_s, loss)

        # 退火：按已完成 epoch 数衰减。λ=0 且 floor=0 时 _anneal_decay() 精确返回 1.0，
        # 下面的乘法会被跳过，故"不退火"不引入任何数值扰动。
        decay = self._anneal_decay()
        if decay != 1.0:
            weight_d = weight_d * decay
            weight_s = weight_s * decay

        loss = loss + loss_d * weight_d + loss_s * weight_s
        if self.tri_bce_log_weights:
            self.last_tri_bce_weights = (float(weight_d.detach().mean()),
                                          float(weight_s.detach().mean()))
        return loss

    # ------------------------------------------------------------------
    # 软权重的两种实现
    # ------------------------------------------------------------------
    def _soft_weight(self, delta, loss_aux, loss_main):
        """给定 Δ = L_aux - L_main，返回软权重。

        softplus：w = alpha · softplus(β·Δ) / β
            β→∞ 时 → alpha·relu(Δ)，正是原版硬截断。
            人话：β 是"硬度旋钮"，β 越大越像原版；我们从 ∞ 拧到有限值。
            用 F.softplus 而不是手写 log(1+exp())，是因为它在 βΔ>20 时自动切到线性分支，
            避免 exp 溢出（手写版在 Δ 稍大时会直接 inf）。
        ratio：w = alpha · Δ / max(L_aux + L_main, eps)
            人话：用"差多少"除以"两个损失加起来多大"，得到量纲无关的相对差距。
            分母的 eps 是纯数值保护（两个损失都趋近 0 时不除零）；
            BCE 恒为正，正常情况下 eps 不起作用。
        """
        if self.tri_bce_mode == "softplus":
            beta = self.tri_bce_beta
            if beta <= 0:
                raise ValueError("tri_bce_beta 必须 > 0（它是有量纲的锐度），收到 %r" % beta)
            w = self.tri_bce_alpha * F.softplus(beta * delta) / beta
        elif self.tri_bce_mode == "smoothmax":
            # Δ · sigmoid(βΔ)：把 relu 的拐角磨圆，且**去掉 softplus 的常数偏置**
            #   Δ→0 时 ≈ Δ/2，Δ→+∞ 时 → Δ（与 relu 同斜率），Δ→-∞ 时 → 0
            beta = self.tri_bce_beta
            if beta <= 0:
                raise ValueError("tri_bce_beta 必须 > 0（它是有量纲的锐度），收到 %r" % beta)
            w = self.tri_bce_alpha * delta * torch.sigmoid(beta * delta)
        elif self.tri_bce_mode == "bsw":
            # BSW-Tri：w = alpha · tanh( beta · relu(Δ) / L_main )
            #
            # 设计取舍（完整证明见 docs/BSW-Tri设计分析.md）：
            #   作者 w = max(0, Δ) 有两个毛病：无上界、单点不可导。
            #   但「Δ≤0 时严格为 0」与「处处可导」在数学上**互不相容**
            #   （0 处左导数为 0，右导数若 >0 则不连续）。
            #   故本方案保留前者（与作者语义一致），只补上「有界」与「尺度归一」：
            #     ① relu(Δ)      → Δ≤0 时权重恒为 0，与作者完全一致
            #     ② tanh(·)      → 权重有界于 alpha，饱和后不再增长
            #     ③ 除以 L_main  → 用相对差而非绝对差，跨数据集/跨训练阶段可比
            #
            # beta 的取法：tanh(x)≈x 当 x→0，故小 Δ 时 w ≈ alpha·beta·Δ/L。
            #   取 beta = 0.45 ≈ Criteo 的 L_0 时，小 Δ 区与作者几乎重合；
            #   取 beta = 1.0 则等价于「直接拿相对差当权重」。beta 作为超参扫描。
            beta = self.tri_bce_beta
            if beta <= 0:
                raise ValueError("tri_bce_beta 必须 > 0（它是有量纲的锐度），收到 %r" % beta)
            positive = torch.clamp(delta, min=0.0)   # 与作者一致：Δ≤0 不奖不罚
            if self.tri_bce_bsw_relative:
                denom = torch.clamp(loss_main, min=self.tri_bce_eps)
                arg = beta * positive / denom
            else:
                # 消融用：分母换成常数（绝对差口径），验证「尺度归一」的作用
                arg = beta * positive
            w = self.tri_bce_alpha * torch.tanh(arg)
        elif self.tri_bce_mode == "ratio":
            denom = torch.clamp(loss_aux + loss_main, min=self.tri_bce_eps)
            w = self.tri_bce_alpha * delta / denom
        else:  # pragma: no cover - 已被 __init__ 拦截
            raise ValueError("未知的 tri_bce_mode: %r" % (self.tri_bce_mode,))
        if self.tri_bce_clamp_nonneg:
            w = torch.clamp(w, min=0.0)
        return w

    def _anneal_decay(self):
        """退火系数：floor + (1-floor)·exp(-λ·k)，k = 已完成的 epoch 数。

        k 从 BaseModel 已有的 `_epoch_index` 读（fit() 每轮更新它），
        **不修改框架任何代码**。取不到时（例如单测直接调 add_loss）按 k=0。
        λ=0 且 floor=0 时精确返回 1.0，故"不退火"对数值零影响。
        """
        lam = self.tri_bce_anneal_lambda
        floor = self.tri_bce_anneal_floor
        if lam == 0.0 and floor == 0.0:
            return 1.0
        k = float(getattr(self, "_epoch_index", 0) or 0)
        return floor + (1.0 - floor) * math.exp(-lam * k)


# ---------------------------------------------------------------------------
# 同名别名：让 run_expid 那套"getattr(model_zoo, params['model'])"无需改动，
# 配置文件里写 model: FCNv2 就落到改进版（默认开关 = 原版行为）。
# 用独立变量而不是 class 重定义，保证 FCNv2Soft is FCNv2，不会出现两个不同的类。
# ---------------------------------------------------------------------------
FCNv2 = FCNv2Soft
