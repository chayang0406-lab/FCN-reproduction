#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LCA 压缩率的参数量推导与核对工具（纯 CPU、秒级、无需数据）。

回答四个问题：
  1. 为什么原实现的 `input_dim // 2` 不是随便写的？（答：cat 后要正好宽 D）
  2. 把 r 参数化之后，为什么必须补一个宽度适配器？
  3. r=1 为什么必须走独立的「全秩」路径（不带 γ/β）？
  4. 各 r 下的理论参数量是多少，实测是否一致？

同时**验证一个候选替代方案是否可行**（flagged as rejected）：
    "H_1 宽 d、H_2 宽 D-d，这样 cat 后天然等于 D" —— 看起来更优雅，但需要
    第二个独立投影 w2: D→(D-d)。本脚本证明：那样 r=1/2 时每层会多出 195,312 个参数
    （195,936 → 391,248，+99.7%），且与原仓库**不可能**逐位一致。
    所以本项目不采用它，而是用零参数宽度适配器。

用法：python code/src/tools_check_lca.py
"""

from __future__ import absolute_import

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.fcn_v2_soft import (                       # noqa: E402
    cross_net_param_count, resolve_projection_dim, is_full_rank,
)

D = 624                # Criteo: 39 字段 × embedding_dim 16
EMB = 14_571_952       # 39 个特征词表 × 16（实测）


def per_layer(D, d, full_rank=False):
    """单层参数量。

    LCA 结构   : w(D×d) + γ(d) + β(d) + b(D) = D·d + 2d + D
    全秩结构   : w(D×d) + b(D)              = D·d + D      （d == D，无 γ/β）
    """
    return D * d + D if full_rank else D * d + 2 * d + D


def layer_params_for(ratio, D=D):
    """按 r 取该档实际使用的单层参数量（自动区分全秩 / LCA）。"""
    if is_full_rank(ratio):
        return per_layer(D, D, full_rank=True)
    return per_layer(D, resolve_projection_dim(D, ratio), full_rank=False)


def alternative_two_projection(D, r):
    """候选方案（已否决）：两个独立投影 w1: D→d, w2: D→(D-d)。"""
    d = D // 2 if abs(r - 0.5) < 1e-12 else int(round(D * r))
    d2 = D - d
    return D * d + D * d2 + 2 * d + 2 * d2 + D


def main():
    ok = True

    print("=" * 78)
    print("1) 原实现为什么固定 //2")
    print("=" * 78)
    d0 = D // 2
    print("   D = %d, d = D//2 = %d" % (D, d0))
    print("   cat([H1, H2]) 宽度 = d + d = %d  == D  -> 残差 x_anchor*(H+b)+x 形状合法"
          % (2 * d0))
    print("   若 d 取别的值（例如 D//4=%d），cat 后宽度 %d != %d -> 加 b（宽 %d）直接报错"
          % (D // 4, 2 * (D // 4), D, D))
    print("   结论：//2 的唯一作用就是让「拼接后宽度 == D」。")
    print("         r 参数化后必须显式补回这个约束（用零参数适配器）。")

    print()
    print("=" * 78)
    print("2) 各压缩率下的参数量（5 层 / 9 层两套口径）")
    print("=" * 78)
    for label, L, E in (("5 层 (exp=4,lin=1, E0 基线)", 5, 4), ("9 层 (exp=4,lin=5, E4-B 主实验)", 9, 4)):
        print()
        print("  %s" % label)
        print("  %-6s %-6s %-10s %-12s %-12s %-12s" %
              ("r", "d", "结构", "单子网", "双子网合计", "模型总参数"))
        base_tot = None
        for r in (0.25, 0.5, 0.75, 1.0):
            full = is_full_rank(r)
            dd = D if full else resolve_projection_dim(D, r)
            one = cross_net_param_count(D, L, 0, r)
            two = 2 * one
            tot = EMB + two
            if abs(r - 0.5) < 1e-12:
                base_tot = tot
            print("  %-6.2f %-6d %-10s %-12s %-12s %-12s" %
                  (r, dd, "全秩" if full else "LCA",
                   "{:,}".format(one), "{:,}".format(two), "{:,}".format(tot)))
        # 比值
        b = cross_net_param_count(D, L, 0, 0.5)
        f = cross_net_param_count(D, L, 0, 1.0)
        print("         比值 r=1 : r=1/2 = %.4f  （不是 2.00，因为 r=1 少 γ/β 的 2D/层）"
              % (f / float(b)))

    print()
    print("   关键提醒：模型总参数里 embedding 占 {:,} / 16,532,562 = {:.1f}%，".format(EMB, 100.0 * EMB / 16_532_562))
    print("   所以交叉网络翻倍只让总量涨 11.74%（5 层）/ 19.30%（9 层）。报告必须分开写两个口径。")

    print()
    print("=" * 78)
    print("3) 候选方案『双投影 w1:D->d, w2:D->(D-d)』为何被否决")
    print("=" * 78)
    print("   该方案 cat 后天然等于 D，不需要适配器，看起来更干净。代价：")
    for r in (0.25, 0.5, 0.75, 1.0):
        mine = layer_params_for(r)
        alt = alternative_two_projection(D, r)
        print("   r=%.2f  本实现每层 %7d   双投影每层 %7d   多出 %7d (%+.1f%%)"
              % (r, mine, alt, alt - mine, 100.0 * (alt / float(mine) - 1)))
    print()
    print("   关键问题在 r=1/2：双投影每层 %d 参数，而原版是 %d —— **结构不同**，"
          % (alternative_two_projection(D, 0.5), layer_params_for(0.5)))
    print("   所以它不可能与原仓库逐位一致，A/B 对照失去基准。")
    print("   >> 结论：保持单一投影 + 零参数适配器。适配器在 r=1/2 时是恒等（零算子），")
    print("      这是「开关关闭 == 原版」这个硬验收成立的技术前提。")

    print()
    print("=" * 78)
    print("4) 自检")
    print("=" * 78)

    # 4.1 r=1/2 必须等于原版的 195,936/层（5 层口径，逐项对账）
    expect_half = D * (D // 2) + 2 * (D // 2) + D
    got_half = layer_params_for(0.5)
    if got_half != expect_half:
        print("   FAIL: r=1/2 每层 %d != 原版 %d" % (got_half, expect_half))
        ok = False
    else:
        print("   PASS: r=1/2 每层 = %d（= 原版 D·d+2d+D，逐项一致）" % got_half)

    # 4.2 r=1/2 模型总参数 == 16,532,562
    t_half = EMB + 2 * cross_net_param_count(D, 5, 0, 0.5)
    if t_half != 16_532_562:
        print("   FAIL: 5 层 r=1/2 模型总参数 %d != 16,532,562" % t_half)
        ok = False
    else:
        print("   PASS: 5 层 r=1/2 模型总参数 = 16,532,562（与 HEAD 实测一致）")

    # 4.3 比值必须约 1.99，且**不是** 2.00
    for L in (5, 9):
        b = cross_net_param_count(D, L, 0, 0.5)
        f = cross_net_param_count(D, L, 0, 1.0)
        ratio = f / float(b)
        if not (1.985 < ratio < 1.995):
            print("   FAIL: %d 层 r=1/r=0.5 = %.4f，应约 1.99（非 2.00）" % (L, ratio))
            ok = False
        else:
            print("   PASS: %d 层 r=1/r=0.5 = %.4f ≈ 1.99（不是 2.00）" % (L, ratio))

    # 4.4 r=1 必须走全秩（无 γ/β）
    lca_formula_at_full = D * D + 2 * D + D      # 若把 LCA 公式代 r=1
    full_rank_layer = layer_params_for(1.0)
    if full_rank_layer != D * D + D:
        print("   FAIL: r=1 每层 %d 不等于全秩的 %d" % (full_rank_layer, D * D + D))
        ok = False
    else:
        print("   PASS: r=1 每层 = %d（全秩，无 γ/β；LCA 公式会给 %d，多 %d）"
              % (full_rank_layer, lca_formula_at_full, lca_formula_at_full - full_rank_layer))

    # 4.5 投影宽度均 > 0
    widths = [resolve_projection_dim(D, r) for r in (0.25, 0.5, 0.75)]
    if any(w <= 0 for w in widths) or resolve_projection_dim(D, 1.0) != D:
        print("   FAIL: 投影宽度异常 %s" % widths)
        ok = False
    else:
        print("   PASS: r in {1/4,1/2,3/4,1} 的投影宽度 %s = %s，均 > 0"
              % ([0.25, 0.5, 0.75, 1.0], widths + [D]))

    # 4.6 与论文 Table 5 四数对账
    checks = [
        ("ECN",       4, 0.5, 0.78),
        ("ECN-full",  4, 1.0, 1.56),
        ("FCNp",      9, 0.5, 1.76),
        ("FCNp-full", 9, 1.0, 3.51),
    ]
    all_hit = True
    for name, L, r, claimed in checks:
        calc_m = cross_net_param_count(D, L, 0, r) / 1e6
        if abs(round(calc_m, 2) - claimed) > 1e-9:
            print("   FAIL: 论文 %s 反解 %.4fM 舍入后 != %.2fM" % (name, calc_m, claimed))
            ok = False
            all_hit = False
    if all_hit:
        print("   PASS: 论文 Table 5 四数全部命中（0.78 / 1.56 / 1.76 / 3.51）")

    print()
    print("总体：%s" % ("全部自检通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
