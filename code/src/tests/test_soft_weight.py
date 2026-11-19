# -*- coding: utf-8 -*-
"""软权重（Tri-BCE）与压缩率（LCA）的**性质测试**。

逐位一致性测试（test_bit_exact.py）只证明"开关关掉时没坏"；
这里证明"开关打开时确实按数学定义在做，而且做得有意义"。分三组：

A. 软权重的数学性质
    A1 softplus 在 β→∞ 时确实收敛到原版的 relu(Δ)
    A2 smoothmax 没有常数偏置（这是与 softplus 的关键区别，实测发现的问题）
    A3 软权重在 Δ=0 附近**梯度不为 0**，而原版硬截断在此处梯度为 0
       —— 这是改动的核心动机，必须被测试固定住
    A4 ratio 模式量纲无关（损失同时放大 k 倍，权重不变）
    A5 退火：λ=0 时严格无影响；λ>0 时随 epoch 单调下降

B. 压缩率的参数量规律
    B1 r=1/2 与 r=1 的交叉网络参数之比 ≈ 2（论文 -full 变体的对应关系）
    B2 解析式与实测参数量完全相等（四个 r 全部对账）
    B3 四个 r 都能正常前反向（形状正确，无死参数）

C. 端到端"能训练"
    C1 每种 tri_bce_mode 都能跑若干步优化，loss 有限、梯度有限
    C2 每种 r 都能跑若干步优化
"""

from __future__ import absolute_import

import copy
import math
import os
import sys
import shutil
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(os.path.dirname(_HERE))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src._orig_loader import orig                        # noqa: E402
from src import fcn_v2_soft                              # noqa: E402
from src.tests import _fixtures as fx                    # noqa: E402
from src.tests.test_bit_exact import _build_pair         # noqa: E402

RATIOS = (0.25, 0.5, 0.75, 1.0)


# =============================================================================
# A. 软权重的数学性质
# =============================================================================
def test_A1_softplus_converges_to_relu():
    """softplus/β 在 β 很大时应收敛到 relu；这是"原版是软权重极端情形"的实证。"""
    deltas = torch.linspace(-0.5, 0.5, 101, dtype=torch.float64)
    relu = torch.clamp(deltas, min=0.0)
    for beta in (1.0, 1e2, 1e4, 1e6):
        approx = torch.nn.functional.softplus(beta * deltas) / beta
        err = float((approx - relu).abs().max())
        # 理论最大误差 = log(2)/β（发生在 Δ=0 处）
        bound = math.log(2.0) / beta * 1.01 + 1e-12
        assert err <= bound, (
            "β=%g 时 softplus 对 relu 的最大误差 %.3e 超过理论上界 %.3e"
            % (beta, err, bound))
    # 反向对照：β=1 时误差应显著（说明"必须调 β"不是空话）
    err_small = float((torch.nn.functional.softplus(1.0 * deltas) / 1.0 - relu).abs().max())
    assert err_small > 0.5, (
        "β=1 时误差只有 %.3e，与预期（约 log2=0.693）不符" % err_small)
    print("    β=1 误差 %.3f，β=1e6 误差 %.2e —— 证实必须调 β" % (err_small, 1e-6 * math.log(2)))


def test_A2_smoothmax_has_no_constant_bias():
    """smoothmax（Δ·sigmoid(βΔ)）在 Δ→0 时趋于 0，而 softplus 趋于 log(2)/β。

    这是本实现在实测中发现的关键差异：Criteo 上 Δ 典型量级 1e-4~1e-3，
    若用 β=1 的 softplus，权重会恒为 ~0.69（比 Δ 大三个数量级），训练直接崩。
    """
    tiny = torch.tensor([0.0, 1e-6, 1e-5, 1e-4], dtype=torch.float64)
    for beta in (1.0, 10.0, 100.0):
        sm = tiny * torch.sigmoid(beta * tiny)
        sp = torch.nn.functional.softplus(beta * tiny) / beta
        assert float(sm[0]) == 0.0, "smoothmax 在 Δ=0 处应为 0，实际 %g" % float(sm[0])
        # smoothmax 在 Δ=1e-4、β=1 时应 ≈ Δ/2 = 5e-5，与 Δ 同量级
        assert float(sm[3]) < 10 * 1e-4, (
            "β=%g 时 smoothmax(1e-4)=%.3e，应与 Δ 同量级" % (beta, float(sm[3])))
        # softplus 在 Δ=0 处恒为 log2/β（常数偏置）
        assert abs(float(sp[0]) - math.log(2.0) / beta) < 1e-12
    print("    Δ=0: smoothmax=0, softplus(β=1)=%.3f —— 常数偏置确认" % math.log(2.0))


def test_A3_soft_weight_has_nonzero_gradient_at_zero():
    """核心动机：Δ=0 附近，原版硬截断梯度为 0，软权重梯度非 0。

    这是"把拐角磨圆"这件事唯一有意义的证明方式——不比数值大小，比梯度。
    """
    eps = 1e-6

    def grad_of(fn, delta_value):
        """在 delta_value 处求 fn(Δ) 对 Δ 的梯度。"""
        p = torch.tensor([delta_value], dtype=torch.float64, requires_grad=True)
        return float(torch.autograd.grad(fn(p).sum(), p)[0])

    # 原版：relu 在负侧的梯度恰好是 0 —— 这就是"死区"
    g_relu_neg = grad_of(lambda d: torch.clamp(d, min=0.0), -eps)
    g_relu_pos = grad_of(lambda d: torch.clamp(d, min=0.0), eps)
    assert g_relu_neg == 0.0, "relu 在 Δ<0 处梯度应为 0（原版的死区）"
    assert g_relu_pos == 1.0, "relu 在 Δ>0 处梯度应为 1"

    # smoothmax：两侧梯度都 > 0
    for beta in (1.0, 10.0):
        g_neg = grad_of(lambda d, b=beta: d * torch.sigmoid(b * d), -eps)
        g_pos = grad_of(lambda d, b=beta: d * torch.sigmoid(b * d), eps)
        assert g_neg > 0.0, (
            "β=%g 时 smoothmax 在 Δ<0 处梯度为 %.3e，应为正（否则死区仍在）"
            % (beta, g_neg))
        assert 0.0 < g_pos < 1.05, (
            "β=%g 时 smoothmax 在 Δ>0 处梯度 %.3f 应在 (0, ~1]" % (beta, g_pos))

    # softplus：同样两侧都 > 0
    g_sp_neg = grad_of(lambda d: torch.nn.functional.softplus(1.0 * d) / 1.0, -eps)
    g_sp_pos = grad_of(lambda d: torch.nn.functional.softplus(1.0 * d) / 1.0, eps)
    assert g_sp_neg > 0.0 and g_sp_pos > 0.0

    # ratio：在 L_aux≈L_main≈0.436 时，梯度 ≈ 1/(2L) ≈ 1.15，也非 0
    L = torch.tensor([0.436], dtype=torch.float64)
    g_ratio_neg = grad_of(
        lambda d: (lambda la: d / torch.clamp(la + (la - d), min=1e-8))(L), -eps)
    assert g_ratio_neg > 0.0, "ratio 模式在 Δ<0 处梯度应为正"

    print("    Δ=-1e-6 处梯度：relu=%.1f（死区）, smoothmax(β=1)=%.3f, softplus(β=1)=%.3f, ratio=%.3f"
          % (g_relu_neg, grad_of(lambda d: d * torch.sigmoid(1.0 * d), -eps), g_sp_neg, g_ratio_neg))


def test_A4_ratio_is_scale_invariant():
    """ratio 模式量纲无关：两个损失同时乘 k，权重不变。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnratio_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=2, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        _, m_new, _fm = _build_pair(tmpdir, dict(mk, tri_bce_mode="ratio"))
        d = torch.tensor([0.001, -0.002, 0.003], dtype=torch.float64)
        la = torch.tensor([0.437, 0.440, 0.435], dtype=torch.float64)
        lm = la - d
        w1 = m_new._soft_weight(d, la, lm)
        w2 = m_new._soft_weight(d * 3.0, la * 3.0, lm * 3.0)
        assert torch.allclose(w1, w2, rtol=0, atol=1e-15), (
            "ratio 模式不满足量纲无关：%s vs %s" % (w1.tolist(), w2.tolist()))
        print("    ratio 在 (Δ,L) 与 3×(Δ,L) 下权重一致：%s" % ("%.6f" % float(w1[0])))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_A5_anneal_behaviour():
    """退火：λ=0 严格无影响；λ>0 时系数随 epoch 单调下降且不低于 floor。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnahead_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=1, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)

        # λ=0, floor=0 → 精确 1.0
        _, m0, _ = _build_pair(tmpdir, dict(mk, tri_bce_mode="smoothmax"))
        assert m0._anneal_decay() == 1.0, "λ=0 且 floor=0 时退火系数必须精确为 1.0"
        m0._epoch_index = 17
        assert m0._anneal_decay() == 1.0, "λ=0 时 epoch 变化不应影响系数"

        # λ>0 → 单调下降
        d2 = tempfile.mkdtemp(prefix="fcnahead2_")
        try:
            _, m1, _ = _build_pair(d2, dict(mk, tri_bce_mode="smoothmax",
                                            tri_bce_anneal_lambda=0.5,
                                            tri_bce_anneal_floor=0.2))
        finally:
            shutil.rmtree(d2, ignore_errors=True)
        vals = []
        for k in range(0, 12):
            m1._epoch_index = k
            vals.append(m1._anneal_decay())
        assert vals[0] == 1.0, "第 0 轮系数应为 1.0，实际 %r" % vals[0]
        assert all(b <= a + 1e-15 for a, b in zip(vals, vals[1:])), "系数非单调下降：%r" % vals
        assert all(v >= 0.2 - 1e-12 for v in vals), "系数低于 floor：%r" % vals
        assert vals[-1] < 0.25, "12 轮后系数应已接近 floor，实际 %.4f" % vals[-1]
        print("    退火系数 k=0..11：%s" % ", ".join("%.3f" % v for v in vals))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# B. 压缩率的参数量规律
# =============================================================================
def test_B1_ratio_one_is_double_of_half():
    """r=1 与 r=1/2 的交叉网络参数之比 ≈ 1.99（不是 2.00，见下）。

    为什么不是精确 2.00：
        r<1 走 LCA 结构，每层 D·d + 2d + D（含 γ/β 两个 d 宽仿射）；
        r=1 走**全秩**结构，每层 D·D + D（无 γ/β，因为维度本就一致、不需要补偿）。
        所以 r=1 每层比"LCA 公式代 r=1"少了 2D 个参数，比值从 2.000 掉到约 1.99。

    这个 1.99 是**对的**，与论文自己的表述一致：
        ECN 0.78 → 1.56 是 2.00×；FCN𝑝 1.76 → 3.51 是 1.99×。
    """
    D = 624
    base = fcn_v2_soft.cross_net_param_count(D, 4, 1, 0.5)   # 5 层
    full = fcn_v2_soft.cross_net_param_count(D, 4, 1, 1.0)   # 5 层，全秩
    ratio = full / float(base)
    assert 1.985 < ratio < 1.995, (
        "5 层 r=1/r=0.5 的交叉网络参数比 = %.4f，应约等于 1.99（不是 2.00）" % ratio)
    print("    5 层交叉网络：r=1/2 → %d, r=1 → %d, 比值 %.4f" % (base, full, ratio))

    # 9 层（对齐论文 FCNp）：1,764,049 / 3,510,625
    b9 = fcn_v2_soft.cross_net_param_count(D, 4, 5, 0.5)
    f9 = fcn_v2_soft.cross_net_param_count(D, 4, 5, 1.0)
    assert b9 == 1_764_049, "9 层 r=1/2 应为 1,764,049，实际 %d" % b9
    assert f9 == 3_510_625, "9 层 r=1 应为 3,510,625，实际 %d" % f9
    assert abs(b9 / 1e6 - 1.76) < 0.005, "9 层 r=1/2 应对齐论文 1.76M"
    assert abs(f9 / 1e6 - 3.51) < 0.005, "9 层 r=1 应对齐论文 3.51M"
    print("    9 层（对齐论文 FCN𝑝）：%.4fM / %.4fM，比值 %.4f —— 论文写 1.76/3.51"
          % (b9 / 1e6, f9 / 1e6, f9 / float(b9)))

    # 4 层（对齐论文 ECN）：784,369 / 1,560,625
    b4 = fcn_v2_soft.cross_net_param_count(D, 4, 0, 0.5)
    f4 = fcn_v2_soft.cross_net_param_count(D, 4, 0, 1.0)
    assert abs(b4 / 1e6 - 0.78) < 0.005, "4 层 r=1/2 应对齐论文 0.78M，实际 %.4fM" % (b4 / 1e6)
    assert abs(f4 / 1e6 - 1.56) < 0.005, "4 层 r=1 应对齐论文 1.56M，实际 %.4fM" % (f4 / 1e6)
    print("    4 层（对齐论文 ECN  ）：%.4fM / %.4fM —— 论文写 0.78/1.56"
          % (b4 / 1e6, f4 / 1e6))

    # 同时诚实报告：模型**总**参数只涨约 11.8%，因为大头是 embedding
    emb_params = sum(vs * 16 for vs in fx.CRITEO_VOCAB.values())
    e2l = fcn_v2_soft.cross_net_param_count(D, 4, 1, 0.5)
    l2e = fcn_v2_soft.cross_net_param_count(D, 1, 4, 0.5)
    total_half = emb_params + e2l + l2e
    e2l_f = fcn_v2_soft.cross_net_param_count(D, 4, 1, 1.0)
    l2e_f = fcn_v2_soft.cross_net_param_count(D, 1, 4, 1.0)
    total_full = emb_params + e2l_f + l2e_f
    growth = total_full / float(total_half) - 1.0
    assert abs(total_half - 16_532_562) <= 2, (
        "解析总参数 %d 与实测 16,532,562 差异过大" % total_half)
    emb_share = emb_params / float(total_half)
    assert 0.87 < emb_share < 0.89, (
        "embedding 占比 %.4f 应在 87%%~89%%" % emb_share)
    print("    模型总参数：r=1/2 → %d, r=1 → %d, 增长 %.2f%%（embedding 占 %.1f%%）"
          % (total_half, total_full, growth * 100, emb_share * 100))


def test_B1b_full_rank_path_has_no_gamma_beta():
    """r=1 必须走全秩路径：state_dict 里**没有** γ/β，且不做 concat。

    这条测试直接盯住"对齐论文 -full 口径"这个决定：
        如果 r=1 还带着 γ/β，参数会多 1,248/层，与论文 0.78→1.56、1.76→3.51 对不上。
    """
    tmpdir = tempfile.mkdtemp(prefix="fcnfull_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)

        d_full = tempfile.mkdtemp(prefix="full_", dir=tmpdir)
        _, m_full, fm = _build_pair(d_full, dict(mk, lca_compress_ratio=1.0))
        keys_full = list(m_full.E2LCN.state_dict().keys())
        assert not any("gamma" in k for k in keys_full), (
            "r=1 的 state_dict 里仍有 gamma：%s" % [k for k in keys_full if "gamma" in k])
        assert not any("beta" in k for k in keys_full), (
            "r=1 的 state_dict 里仍有 beta")
        assert m_full.E2LCN.full_rank is True
        assert m_full.E2LCN.gamma is not None and len(m_full.E2LCN.gamma) == 0, \
            "r=1 时 gamma 列表应为空"

        # r=1/2 必须有 gamma/beta
        d_half = tempfile.mkdtemp(prefix="half_", dir=tmpdir)
        _, m_half, _ = _build_pair(d_half, dict(mk, lca_compress_ratio=0.5))
        keys_half = list(m_half.E2LCN.state_dict().keys())
        assert any("gamma" in k for k in keys_half), "r=1/2 应含 gamma"
        assert m_half.E2LCN.full_rank is False

        # 每层参数差：r=1 每层 D·D+D = 390,000；LCA公式代 r=1 是 D·D+2D+D = 391,248
        print("    r=1   state_dict 无 γ/β（%d 个键），full_rank=True" % len(keys_full))
        print("    r=1/2 含 γ/β（%d 个键），full_rank=False" % len(keys_half))
        print("    差 2D=1,248/层 × 5 层 = 6,240 —— 正是比值从 2.00 变 1.99 的来源")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_B2_analytic_matches_measured_all_ratios():
    """四个 r 的解析参数量都必须与实测完全相等 —— 证明没有隐藏参数。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnratio2_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.1, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        D = 624
        for r in RATIOS:
            d_ = tempfile.mkdtemp(prefix="fcnr_", dir=tmpdir)
            _, m, _fm = _build_pair(d_, dict(mk, lca_compress_ratio=r))
            measured = sum(p.numel() for p in m.E2LCN.parameters())
            analytic = fcn_v2_soft.cross_net_param_count(D, 4, 1, r)
            assert measured == analytic, (
                "r=%.2f 时 E2L 实测 %d 与解析 %d 不符" % (r, measured, analytic))
        print("    四个 r 的解析=实测参数量全部对账通过")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_B3_all_ratios_forward_backward():
    """四个 r 都能正常前反向，所有参数都拿到非零有限梯度（无死参数）。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnratio3_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        for r in RATIOS:
            d_ = tempfile.mkdtemp(prefix="fcnr_", dir=tmpdir)
            _, m, fm = _build_pair(d_, dict(mk, lca_compress_ratio=r))
            m.train()
            batch = fx.make_batch(fm, batch_size=32, seed=5)
            loss = m.add_loss(batch)
            assert torch.isfinite(loss), "r=%.2f 时 loss 非有限" % r
            loss.backward()
            n_total = n_grad = n_nonzero = 0
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                n_total += 1
                assert p.grad is not None, "r=%.2f 参数 %s 无梯度" % (r, name)
                assert torch.isfinite(p.grad).all(), "r=%.2f 参数 %s 梯度非有限" % (r, name)
                n_grad += 1
                if float(p.grad.abs().sum()) > 0:
                    n_nonzero += 1
            assert n_grad == n_total, "r=%.2f 有参数未拿到梯度" % r
            assert n_nonzero == n_total, (
                "r=%.2f 有 %d/%d 个参数梯度全为 0（死参数）"
                % (r, n_total - n_nonzero, n_total))
            print("    r=%.2f: loss=%.6f, %d/%d 参数梯度非零且有限"
                  % (r, float(loss), n_nonzero, n_total))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# C. 端到端"能训练"
# =============================================================================
def _train_a_few_steps(model, batch, steps=5, lr=1e-2):
    """用固定的一个 batch 跑几步 Adam，返回 loss 序列（用于验证"确实在学"）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.add_loss(batch)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def test_C1_all_tri_bce_modes_train():
    """每种 tri_bce_mode 都能训练，且 loss 单调性合理（不要求每步下降，但末值应低于首值）。"""
    tmpdir = tempfile.mkdtemp(prefix="fcntrain_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=2, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-2)
        for mode in fcn_v2_soft.TRI_BCE_MODES:
            d_ = tempfile.mkdtemp(prefix="fcnm_", dir=tmpdir)
            cfg = dict(mk, tri_bce_mode=mode)
            if mode == "smoothmax":
                cfg["tri_bce_beta"] = 100.0     # 与 Δ 的量级匹配（见文档说明）
            elif mode == "softplus":
                cfg["tri_bce_beta"] = 1000.0    # 同上，否则常数偏置会主导
            _, m, fm = _build_pair(d_, cfg)
            batch = fx.make_batch(fm, batch_size=64, seed=9)
            losses = _train_a_few_steps(m, batch, steps=8)
            assert all(math.isfinite(x) for x in losses), (
                "mode=%s 出现非有限 loss：%r" % (mode, losses))
            assert losses[-1] < losses[0], (
                "mode=%s 训练 8 步后 loss 未下降：%.6f → %.6f"
                % (mode, losses[0], losses[-1]))
            print("    %-10s loss %.4f → %.4f" % (mode, losses[0], losses[-1]))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_C2_all_ratios_train():
    """每个压缩率都能训练（形状正确、优化可走）。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnrtrain_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=2, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-2)
        for r in RATIOS:
            d_ = tempfile.mkdtemp(prefix="fcnr_", dir=tmpdir)
            _, m, fm = _build_pair(d_, dict(mk, lca_compress_ratio=r))
            batch = fx.make_batch(fm, batch_size=64, seed=9)
            losses = _train_a_few_steps(m, batch, steps=8)
            assert all(math.isfinite(x) for x in losses), (
                "r=%.2f 出现非有限 loss：%r" % (r, losses))
            assert losses[-1] < losses[0], (
                "r=%.2f 训练 8 步后 loss 未下降" % r)
            print("    r=%.2f   loss %.4f → %.4f" % (r, losses[0], losses[-1]))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_C3_bad_config_raises():
    """开关拼错必须立刻报错，不能静默退回原版（静默退回会让 A/B 变成假对照）。"""
    tmpdir = tempfile.mkdtemp(prefix="fcnbad_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=1, lin_num_layers=1,
                  net_dropout=0.0, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        # 拼错的 mode
        try:
            _build_pair(tmpdir, dict(mk, tri_bce_mode="softpluss"))
            raise AssertionError("拼错的 tri_bce_mode='softpluss' 没有报错")
        except ValueError:
            pass
        # 越界的 r
        for bad_r in (0.0, -0.5, 1.5, 2.0):
            try:
                _build_pair(tmpdir, dict(mk, lca_compress_ratio=bad_r))
                raise AssertionError("越界 lca_compress_ratio=%r 没有报错" % bad_r)
            except ValueError:
                pass
        # β<=0
        try:
            _build_pair(tmpdir, dict(mk, tri_bce_mode="softplus", tri_bce_beta=0.0))
        except ValueError:
            pass
        else:
            # β 校验在 _soft_weight 里，构造阶段不触发；手动触发一次
            d2 = tempfile.mkdtemp(prefix="fcnbad2_", dir=tmpdir)
            _, m, fm = _build_pair(d2, dict(mk, tri_bce_mode="softplus", tri_bce_beta=0.0))
            try:
                m.add_loss(fx.make_batch(fm, batch_size=8, seed=1))
                raise AssertionError("tri_bce_beta=0 时 add_loss 没有报错")
            except ValueError:
                pass
        print("    非法开关全部被拦截")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s\n      %s" % (t.__name__, e))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR %s\n      %r" % (t.__name__, e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
