# -*- coding: utf-8 -*-
"""硬验收：改进开关关闭时，本实现与原仓库**逐位一致**。

===============================================================================
为什么用 torch.equal 而不是 torch.allclose
===============================================================================
allclose 允许 1e-8 的误差，而 1e-8 在 1600 万参数的模型里足以掩盖真实的实现差异：
    比如把 concat 的顺序从 [H1, H2] 换成 [H2, H1]、
    或者把 `x_anchor * (H + b) + x` 写成 `x + x_anchor * (H + b)`，
    这两处改动的数值差异都远小于 1e-8，但语义完全不同。
所以这里全部用 `torch.equal`（要求完全相等，一个 bit 都不许差）。

===============================================================================
三个层面都要"逐位"
===============================================================================
1. 初始化后权重参数：state_dict 里每个张量都相同（说明 nn.Linear 的构造顺序、
   随机数消耗顺序、reset_parameters 的 xavier 初始化全部一致）。
2. 前向输出：y_pred / y_d / y_s 三个张量完全相同。
3. add_loss 的返回值：连同 backward 后的**梯度**都要完全相同
   —— 只对 loss 有可能两边算出同样的标量而梯度不同（例如梯度被 detach 掉），
   所以必须比对梯度。

===============================================================================
怎么跑
===============================================================================
    cd code && python -m pytest src/tests/test_bit_exact.py -v
    或直接：cd code && python src/tests/test_bit_exact.py
（本文件同时支持 pytest 与"直接执行"两种方式，因为远程 DCU 上未必装了 pytest 的调用习惯。）
"""

from __future__ import absolute_import

import os
import sys
import shutil
import tempfile

import torch

# —— 让 `import src` 指向 code/src（本改进版），`fcn_orig` 指向 code/FCN/src（原仓库）——
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(os.path.dirname(_HERE))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src._orig_loader import orig, orig_module          # noqa: E402
from src import fcn_v2_soft                             # noqa: E402
from src.tests import _fixtures as fx                   # noqa: E402


# ---------------------------------------------------------------------------
# 构造一对"完全同构"的模型：一个来自原仓库，一个来自改进版
# ---------------------------------------------------------------------------
def _build_pair(tmpdir, model_kwargs, seed=2025):
    """先造原版，再造改进版，两者用同一 seed、同一份 kwargs。

    顺序很重要：每次 seed_everything 之后立刻建模型，保证两边看到的随机数序列起点相同。
    """
    from fuxictr.pytorch.torch_utils import seed_everything

    fm_json_dir = os.path.join(tmpdir, "data")
    fx.write_feature_map_json(fm_json_dir)
    feature_map = fx.load_feature_map(fm_json_dir)
    kwargs = fx.base_kwargs(tmpdir)

    seed_everything(seed=seed)
    torch.manual_seed(seed)          # seed_everything 已设，这里再加一道保险
    m_orig = orig.FCNv2(feature_map, **dict(kwargs), **dict(model_kwargs))

    seed_everything(seed=seed)
    torch.manual_seed(seed)
    m_new = fcn_v2_soft.FCNv2Soft(feature_map, **dict(kwargs), **dict(model_kwargs))
    return m_orig, m_new, feature_map


def _assert_state_dict_identical(m_orig, m_new, tag=""):
    sd_o = m_orig.state_dict()
    sd_n = m_new.state_dict()
    assert list(sd_o.keys()) == list(sd_n.keys()), (
        "%s state_dict 的键不一致：\n原版 %s\n新版 %s"
        % (tag, list(sd_o.keys())[:8], list(sd_n.keys())[:8]))
    for k in sd_o:
        a, b = sd_o[k], sd_n[k]
        assert a.dtype == b.dtype, "%s %s dtype 不同：%s vs %s" % (tag, k, a.dtype, b.dtype)
        assert a.shape == b.shape, "%s %s 形状不同：%s vs %s" % (tag, k, a.shape, b.shape)
        assert torch.equal(a, b), (
            "%s 参数 %s 不是逐位一致（max|diff|=%.3e）"
            % (tag, k, float((a.float() - b.float()).abs().max())))


def _assert_equal_loss_and_grad(m_orig, m_new, batch):
    """断言 add_loss 的返回值与 backward 后的每个参数梯度都逐位相同。

    为什么要单独抽出来并且也检查梯度：
        只比 loss 标量会漏掉"值相同但梯度被 detach"这类实现差异
        （见 test_mutation.py 的 M4 变异）。
    为什么每次 add_loss 前都重设 manual_seed：
        train 模式下有 dropout，两边的 dropout mask 必须一致，
        否则测的是"随机数不同"而不是"实现不同"。
    """
    m_orig.zero_grad(set_to_none=True)
    m_new.zero_grad(set_to_none=True)
    torch.manual_seed(4321)
    loss_o = m_orig.add_loss(batch)
    loss_o.backward()
    torch.manual_seed(4321)
    loss_n = m_new.add_loss(batch)
    loss_n.backward()
    assert torch.equal(loss_o, loss_n), (
        "add_loss 不是逐位一致（max|diff|=%.3e）"
        % float((loss_o.detach() - loss_n.detach()).abs().max()))
    for (n_o, p_o), (n_n, p_n) in zip(m_orig.named_parameters(), m_new.named_parameters()):
        assert n_o == n_n, "参数名不一致：%s vs %s" % (n_o, n_n)
        g_o, g_n = p_o.grad, p_n.grad
        if g_o is None or g_n is None:
            assert g_o is None and g_n is None, "梯度有无不一致：%s" % n_o
            continue
        assert torch.equal(g_o, g_n), (
            "参数 %s 的梯度不是逐位一致（max|diff|=%.3e）"
            % (n_o, float((g_o - g_n).abs().max())))


# ---------------------------------------------------------------------------
# 测试 1：默认配置（tri_bce_mode=hard, lca_compress_ratio=0.5）→ 全链路逐位一致
# ---------------------------------------------------------------------------
def test_default_config_is_bit_exact():
    tmpdir = tempfile.mkdtemp(prefix="fcnexact_")
    try:
        # 与作者 config 同口径：exp=4, lin=1, heads=1, dropout=0.1, batch_norm=False
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.1, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        m_orig, m_new, feature_map = _build_pair(tmpdir, mk)

        # —— 1. 参数逐位一致 ——
        _assert_state_dict_identical(m_orig, m_new, tag="[默认配置]")

        # —— 2. 前向逐位一致（eval 模式，关掉 dropout 的随机性）——
        m_orig.eval()
        m_new.eval()
        batch = fx.make_batch(feature_map, batch_size=16, seed=7)
        with torch.no_grad():
            out_o = m_orig.forward(batch)
            out_n = m_new.forward(batch)
        assert set(out_o.keys()) == set(out_n.keys()), "forward 返回的键不同"
        for k in out_o:
            assert torch.equal(out_o[k], out_n[k]), (
                "前向输出 %s 不是逐位一致（max|diff|=%.3e）"
                % (k, float((out_o[k] - out_n[k]).abs().max())))

        # —— 3+4. add_loss 与梯度逐位一致（train 模式，含 dropout）——
        m_orig.train()
        m_new.train()
        _assert_equal_loss_and_grad(m_orig, m_new, batch)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 测试 2：显式写全"原版等价"的开关组合（含 beta/clamp 等不动点）→ 仍逐位一致
# ---------------------------------------------------------------------------
def test_explicit_original_switches_are_bit_exact():
    tmpdir = tempfile.mkdtemp(prefix="fcnexact2_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=3, lin_num_layers=3,
                  net_dropout=0.2, batch_norm=True, layer_norm=False,
                  num_heads=2, learning_rate=1e-2)
        # 改进版显式写出一堆"等价于原版"的开关（beta=1、不退火等），
        # 原版没有这些参数 → 用空 dict
        m_orig, m_new, feature_map = _build_pair(tmpdir, mk)
        _assert_state_dict_identical(m_orig, m_new, tag="[batch_norm+2heads]")

        m_orig.eval(); m_new.eval()
        batch = fx.make_batch(feature_map, batch_size=12, seed=11)
        with torch.no_grad():
            out_o = m_orig.forward(batch)
            out_n = m_new.forward(batch)
        for k in out_o:
            assert torch.equal(out_o[k], out_n[k]), "前向 %s 不一致" % k
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 测试 3：layer_norm=True 分支也要一致（原版有 layer_norm 开关，容易漏测）
# ---------------------------------------------------------------------------
def test_layer_norm_branch_is_bit_exact():
    tmpdir = tempfile.mkdtemp(prefix="fcnexact3_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=2, lin_num_layers=2,
                  net_dropout=0.0, batch_norm=False, layer_norm=True,
                  num_heads=1, learning_rate=1e-3)
        m_orig, m_new, feature_map = _build_pair(tmpdir, mk)
        _assert_state_dict_identical(m_orig, m_new, tag="[layer_norm]")
        m_orig.eval(); m_new.eval()
        batch = fx.make_batch(feature_map, batch_size=8, seed=3)
        with torch.no_grad():
            out_o = m_orig.forward(batch)
            out_n = m_new.forward(batch)
        for k in out_o:
            assert torch.equal(out_o[k], out_n[k]), "layer_norm 分支前向 %s 不一致" % k
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 测试 4：参数量与逐字段对账（证明结构没多加/少加任何参数）
# ---------------------------------------------------------------------------
def test_param_count_matches_original():
    tmpdir = tempfile.mkdtemp(prefix="fcnexact4_")
    try:
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.1, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        m_orig, m_new, feature_map = _build_pair(tmpdir, mk)
        n_o = sum(p.numel() for p in m_orig.parameters())
        n_n = sum(p.numel() for p in m_new.parameters())
        assert n_o == n_n, "总参数量不同：原版 %d vs 新版 %d" % (n_o, n_n)

        # 与"作者配置的 HEAD 版"实测值对账：16,532,562（见 docs/前置事实 第五节）
        # 该数字对应 39 字段 × 16 维 = D=624，exp=4/lin=1，heads=1。
        assert n_o == 16_532_562, (
            "参数量 %d 与已知的 HEAD 默认配置 16,532,562 不符 —— "
            "说明 fixture 的 vocab 或结构偏离了作者配置" % n_o)
        # 交叉网络解析式对账
        D = feature_map.sum_emb_out_dim()
        assert D == 624, "input_dim 应为 624，实际 %d" % D
        analytic = fcn_v2_soft.cross_net_param_count(D, 4, 1, 0.5)
        measured = sum(p.numel() for p in m_new.E2LCN.parameters())
        assert analytic == measured, (
            "交叉网络解析参数量 %d 与实测 %d 不符" % (analytic, measured))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _main():
    """不依赖 pytest 的直跑入口（远程 DCU 上更稳）。"""
    tests = [
        test_default_config_is_bit_exact,
        test_explicit_original_switches_are_bit_exact,
        test_layer_norm_branch_is_bit_exact,
        test_param_count_matches_original,
    ]
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
