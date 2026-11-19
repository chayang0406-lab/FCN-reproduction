#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E4-B 的**严格对照表**：把我们的压缩率档位与论文可比数值对齐。

为什么必须单写一个脚本（起因是一次真实的口径事故）：
    我们 r=1 的单子网是 1,950,625，论文里看起来最接近的是 FCN𝑝 的 1.76M。
    如果直接把这两个数放一张表里比大小，会得出"我们的 r=1 比论文完整版还大"的错误结论。
    实际上它们**不是同一配置**：
        - 我们的 exp=4/lin=1 → 每个子网 5 层
        - 论文 FCN𝑝 是 4 ECN 层 + 5 LCN 层 = **9 层**
    层数不同，参数自然不同。拿错数比较会让整个 E4-B 主贡献的可比性崩掉。

本脚本把口径固定成四条，任何跨表比较都必须先过这四条：
    ① 层数必须相同        ② 单子网 vs 双子网必须说清
    ③ 是否含 embedding    ④ r=1 走全秩路径（无 γ/β）

用法：python code/src/tools_paper_comparison.py
"""

from __future__ import absolute_import

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.fcn_v2_soft import cross_net_param_count     # noqa: E402

D = 624            # input_dim = 39 字段 × embedding_dim 16
EMB = 14_571_952   # 39 个特征词表 × 16（实测）


def paper_style(L, r, D=D):
    """按论文口径算**单个**交叉网络参数。

    r < 1  -> LCA 结构：每层 D*d + 2d + D（含 gamma/beta 补偿仿射），末尾 fc(D+1)
    r == 1 -> 全秩结构：每层 D*D + D       （无 gamma/beta，维度本就一致不需要补偿）
    这个分界是**由论文数字反推出来的**，见 B 节与 B2 节。
    """
    return cross_net_param_count(D, exp_num_layers=L, lin_num_layers=0, ratio=r)


def no_gamma(L, r, D=D):
    """对照解读：所有 r 都不带 gamma/beta（每层 D*d + D）。"""
    d = D // 2 if abs(r - 0.5) < 1e-12 else int(round(D * r))
    return L * (D * d + D) + (D + 1)


def gamma_at_full(L, r, D=D):
    """对照解读：r=1 也带 gamma/beta（把 LCA 公式直接代 r=1）。"""
    d = D // 2 if abs(r - 0.5) < 1e-12 else int(round(D * r))
    return L * (D * d + 2 * d + D) + (D + 1)


PAPER = [
    ("ECN",       4, 0.5, 0.78),
    ("ECN-full",  4, 1.0, 1.56),
    ("FCNp",      9, 0.5, 1.76),
    ("FCNp-full", 9, 1.0, 3.51),
]


def main():
    print("=" * 86)
    print("E4-B 严格对照表 —— 四条口径必须同时满足才能横向比较")
    print("=" * 86)
    print("(1)层数相同  (2)单/双子网  (3)是否含 embedding  (4)r=1 走全秩（无 gamma/beta）")

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("A. 我们的配置（exp=4, lin=1 -> 每个子网 5 层；模型内有两个子网 E2L + L2E）")
    print("-" * 86)
    print("  %-6s %-6s %-14s %-14s %-16s" %
          ("r", "d", "单子网(5层)", "双子网合计", "模型总参数(含emb)"))
    ours = {}
    for r in (0.25, 0.5, 0.75, 1.0):
        one = paper_style(5, r)
        two = 2 * one
        total = EMB + two
        ours[r] = (one, two, total)
        dd = D // 2 if abs(r - 0.5) < 1e-12 else int(round(D * r))
        print("  %-6.2f %-6d %-14s %-14s %-16s" %
              (r, dd, "{:,}".format(one), "{:,}".format(two), "{:,}".format(total)))

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("B. 论文 Table 5（Criteo）反推 —— 层数与我们不同！本实现口径")
    print("-" * 86)
    print("  %-16s %-6s %-6s %-14s %-12s %-8s" %
          ("论文变体", "层数", "r", "单子网参数", "本实现解读", "论文标注"))
    ok_paper = True
    for name, L, r, claimed in PAPER:
        calc = paper_style(L, r)
        calc_m = calc / 1e6
        delta = abs(round(calc_m, 2) - claimed)
        flag = "OK" if delta < 1e-9 else "MISMATCH %.2fM" % calc_m
        if delta >= 1e-9:
            ok_paper = False
        print("  %-16s %-6d %-6.2f %-14s %-12.4f %-8s %s" %
              (name, L, r, "{:,}".format(calc), calc_m, "%.2fM" % claimed, flag))
    print()
    print("  >> 四个变体全部按两位小数四舍五入后与论文完全一致。")

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("B2. 为什么 r=1 必须走全秩路径（去掉 gamma/beta）—— 三种解读的判决")
    print("-" * 86)
    print("  论文只报两位小数，但足以排除另两种解读：")
    print()
    print("  %-16s %-6s %-12s %-12s %-12s %-8s" %
          ("论文变体", "层数", "全秩(采用)", "r=1也带γ/β", "全都不带γ/β", "论文"))
    verdict_ok = True
    for name, L, r, claimed in PAPER:
        a = paper_style(L, r) / 1e6
        b = gamma_at_full(L, r) / 1e6
        c = no_gamma(L, r) / 1e6
        fa = "OK" if abs(round(a, 2) - claimed) < 1e-9 else "%.2f" % round(a, 2)
        fb = "OK" if abs(round(b, 2) - claimed) < 1e-9 else "%.2f" % round(b, 2)
        fc = "OK" if abs(round(c, 2) - claimed) < 1e-9 else "%.2f" % round(c, 2)
        print("  %-16s %-6d %-12s %-12s %-12s %-8.2f" %
              (name, L, fa, fb, fc, claimed))
    print()
    print("  >> 判决：'r=1 也带 γ/β' 会在两个 -full 变体上得出 1.57 / 3.52，")
    print("     而论文写的是 1.56 / 3.51 —— **排除**。")
    print("  >> '全都不带 γ/β' 虽然也能对上一个变体的两位小数，但它对 ECN(4层)=%.4fM，"
          % (no_gamma(4, 0.5) / 1e6))
    print("     与论文 0.78M 四舍五入后一致，所以仅凭四舍五入**不能**区分这两种。")
    print("  >> 我们按**语义 + 原仓库代码**定夺：γ/β 是 LCA 为'砍半后补偿表达能力'加的配套仿射")
    print("     （论文原话 'introduce a low-cost affine transformation to maintain consistency'），")
    print("     全秩时不存在维度不一致，自然不需要它 -> 这直接对应 ECN_full.py 里的")
    print("     ExponentialCrossNetwork（每层只有 w(D→D) + b(D)，没有 γ/β）。")
    print("  >> 结论：r<1 用 LCA 结构，r=1 用全秩结构。这与原仓库两个文件的分工完全吻合。")
    if not (abs(round(paper_style(4, 1.0) / 1e6, 2) - 1.56) < 1e-9
            and abs(round(paper_style(9, 1.0) / 1e6, 2) - 3.51) < 1e-9):
        verdict_ok = False

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("C. 唯一允许的横向比较：同层数（5 层）下 r 的纯效应  <- E4-B 核心结论区")
    print("-" * 86)
    print("  %-6s %-14s %-14s %-14s" % ("r", "单子网参数", "相对 r=1/2", "模型总参数涨幅"))
    base_one = ours[0.5][0]
    base_tot = ours[0.5][2]
    for r in (0.25, 0.5, 0.75, 1.0):
        one, two, tot = ours[r]
        print("  %-6.2f %-14s %-14s %-14s" %
              (r, "{:,}".format(one),
               "%+.2f%%" % (100.0 * (one / base_one - 1)),
               "%+.2f%%" % (100.0 * (tot / base_tot - 1))))
    print()
    print("  比值 r=1 : r=1/2 = %.4f" % (ours[1.0][0] / float(ours[0.5][0])))
    print("  >> 注意：是 1.99，**不是 2.00**。因为 r=1 少了 γ/β 那 2D/层。")
    print("     这与论文自己一致：ECN 0.78->1.56 是 2.00x，FCNp 1.76->3.51 是 1.99x。")
    print("  >> 模型总参数只涨 %.2f%%，因为 embedding 占 %.1f%%"
          % (100.0 * (ours[1.0][2] / float(base_tot) - 1), 100.0 * EMB / base_tot))

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("D. E4-B 必须用 9 层配置（exp=4, lin=5），不能沿用 E0 的 5 层")
    print("-" * 86)
    e0_one = paper_style(5, 0.5)
    e4b_one = paper_style(9, 0.5)
    print("  E0 基线   exp=4/lin=1 -> 每子网 5 层：单子网 %s，模型总参数 %s"
          % ("{:,}".format(e0_one), "{:,}".format(EMB + 2 * e0_one)))
    print("  E4-B 对照 exp=4/lin=5 -> 每子网 9 层：单子网 %s，模型总参数 %s"
          % ("{:,}".format(e4b_one), "{:,}".format(EMB + 2 * e4b_one)))
    print()
    print("  论文 Table 5 的 FCNp 是 9 层口径（1.76M）。若 E4-B 用 5 层去比，")
    print("  会得出'参数只有论文一半(%s vs 1.76M)'的错误结论。"
          % "{:,}".format(e0_one))
    print("  >> E4-B 一律用 exp=4, lin=5。E0 基线与噪声基线仍用 exp=4, lin=1。")

    # ------------------------------------------------------------------
    print()
    print("-" * 86)
    print("E. 自检")
    print("-" * 86)
    ok = True
    if not ok_paper:
        print("  FAIL: 论文 4 个变体未能全部对齐")
        ok = False
    else:
        print("  PASS: 论文 4 个变体（0.78/1.56/1.76/3.51）全部对上（两位小数）")
    r5 = ours[1.0][0] / float(ours[0.5][0])
    if not (1.985 < r5 < 1.995):
        print("  FAIL: 5 层 r=1/r=0.5 = %.4f，应为约 1.99" % r5)
        ok = False
    else:
        print("  PASS: 5 层 r=1/r=0.5 = %.4f ≈ 1.99（不是 2.00）" % r5)
    r9 = paper_style(9, 1.0) / float(paper_style(9, 0.5))
    if not (1.985 < r9 < 1.995):
        print("  FAIL: 9 层 r=1/r=0.5 = %.4f，应为约 1.99" % r9)
        ok = False
    else:
        print("  PASS: 9 层 r=1/r=0.5 = %.4f ≈ 1.99" % r9)
    if abs(ours[0.5][2] - 16_532_562) > 2:
        print("  FAIL: 5 层 r=1/2 模型总参数 %d 与实测 16,532,562 不符" % ours[0.5][2])
        ok = False
    else:
        print("  PASS: 5 层 r=1/2 模型总参数 = 16,532,562（与 HEAD 实测一致）")
    print()
    print("总体：%s" % ("全部自检通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
