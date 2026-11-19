# -*- coding: utf-8 -*-
"""独立复算 `docs/改进方案.md` §4.1 的配对设计功率分析数字。

为什么要单独复算：
    文档里写了「分辨 0.0005 需 32 对种子」这类数字，如果算错了，
    就会得出「改进不可行」的错误结论（或反之，白烧 80 小时算力）。
    这类数字必须能被独立复核，不能只凭手算。

配对面设计（同一 seed 下跑 baseline 与改进版，取 AUC 差值）的样本量公式：
    n = (z_{1-α/2} + z_{1-β})² / d²,   d = Δ/σ
α=0.05 双侧 → z=1.959964；power=0.80 → z=0.841621

用法：python code/src/tools_power_analysis.py
"""

from __future__ import absolute_import

import math
import os
import sys

# 标准正态分位数（避免依赖 scipy）
Z_ALPHA_2 = 1.959963984540054     # Φ⁻¹(1 - 0.05/2)
Z_POWER = 0.8416212335729143      # Φ⁻¹(0.80)

SIGMA = 0.001                     # 种子间 AUC 标准差（配对差值的 σ，需实测确认）

# 探针效应量与文档中的表
TARGETS = [
    (0.0005, 32),
    (0.0008, 13),
    (0.0010, 8),
]


def pairs_needed(delta, sigma=SIGMA, alpha_z=Z_ALPHA_2, power_z=Z_POWER):
    """返回达到给定效应量所需的配对数（向上取整）。"""
    d = delta / sigma
    n = (alpha_z + power_z) ** 2 / (d * d)
    return int(math.ceil(n))


# 配对设计：一对 = 基线一次 + 改进一次 = **2 次训练**。
# 这个常量必须显式存在，任何涉及"配对数 → 训练小时"的换算都必须乘它。
PAIR_TO_RUNS = 2


def training_hours(pairs, minutes_per_run, pair_to_runs=PAIR_TO_RUNS):
    """把配对数换算成训练小时数。

    单独写成函数（而不是内联乘法）是为了让"一对 = 2 次训练"这个口径
    无法在调用处被悄悄漏掉——本项目的假纠错就是这么发生的。
    """
    return pairs * pair_to_runs * minutes_per_run / 60.0


def main():
    ok = True
    print("=" * 76)
    print("配对设计功率分析复算（α=0.05 双侧, power=0.80, σ=%.4g）" % SIGMA)
    print("=" * 76)
    print("  %-10s %-10s %-12s %-10s %-8s" %
          ("Δ(AUC)", "Cohen's d", "公式值 n", "文档写", "判定"))
    for delta, doc_n in TARGETS:
        d = delta / SIGMA
        raw = (Z_ALPHA_2 + Z_POWER) ** 2 / (d * d)
        n = pairs_needed(delta)
        flag = "OK" if n == doc_n else "FAIL(应%d)" % n
        if n != doc_n:
            ok = False
        print("  %-10.4f %-10.4f %-12.2f %-10d %-8s" % (delta, d, raw, doc_n, flag))

    print()
    print("  说明：公式值 31.40 → 取整 32；12.26 → 13；7.85 → 8。文档一致。")

    # 反向：给定配对数能分辨的最小效应
    print()
    print("-" * 76)
    print("反向查表：给定配对数能分辨的最小 Δ")
    print("-" * 76)
    print("  %-10s %-12s %-12s" % ("配对数", "最小可辨 Δ", "折合 AUC 差(pp)"))
    for n in (5, 7, 8, 13, 32):
        d = (Z_ALPHA_2 + Z_POWER) / math.sqrt(n)
        delta = d * SIGMA
        print("  %-10d %-12.6f %-12.3f" % (n, delta, delta * 100))

    print()
    print("  >> 5~7 对（本项目可承受范围）只能分辨 Δ ≈ %.5f（%.2f pp）。"
          % ((Z_ALPHA_2 + Z_POWER) / math.sqrt(6) * SIGMA,
             (Z_ALPHA_2 + Z_POWER) / math.sqrt(6) * SIGMA * 100))
    print("     而预期效应量 ≤ 0.0005（0.05 pp）—— 确实测不出。这是诚实的结论，非失败。")

    # 算力换算
    #
    # ⚠️⚠️ 这里有一个极易错的口径，曾导致本项目一次「假纠错」（见文末教训）：
    #     配对设计里 **一对 = 2 次训练**（baseline 一次 + improved 一次），
    #     所以「32 对」对应 **64 次训练**，不是 32 次。
    #     漏掉这个 ×2，会算出 40.5 小时，把「32 对」误当成「32 次」，
    #     进而得出「32 对只要 40 小时就够」的错误结论（实际翻倍）。
    print()
    print("-" * 76)
    print("算力换算（⚠️ 一对 = %d 次训练；单次耗时受机器负载影响，只能报量级）"
          % PAIR_TO_RUNS)
    print("-" * 76)
    print("  %-8s %-10s %-12s %-12s" % ("配对数", "训练次数", "76 min/次", "150 min/次"))
    for n in (8, 13, 32):
        runs = n * PAIR_TO_RUNS
        print("  %-8d %-10d %-12.1f %-12.1f"
              % (n, runs, training_hours(n, 76), training_hours(n, 150)))
    print()
    runs32 = 32 * PAIR_TO_RUNS
    h32_lo = training_hours(32, 76)
    h32_hi = training_hours(32, 150)
    print("  >> 32 对 = %d 次训练：76 min/次 → **%.1f 小时**；150 min/次 → %.1f 小时。"
          % (runs32, h32_lo, h32_hi))
    print("     **下限是 %.1f 小时，不是 %.1f** —— %.1f 只是 32 次（=16 对）的量。"
          % (h32_lo, h32_lo / PAIR_TO_RUNS, h32_lo / PAIR_TO_RUNS))
    print("  >> ⚠️ 该耗时随 DCU 机器负载显著变化（有第三方进程占用 6 核配额），")
    print("     故报告里建议写「%.0f~%.0f 小时（约 3~7 天）」，并注明随负载浮动。"
          % (h32_lo, h32_hi))

    # —— 硬自检：这三行必须成立，否则说明换算口径又错了 ——
    print()
    print("-" * 76)
    print("算力换算自检（防止再次漏掉「一对 = 2 次」）")
    print("-" * 76)
    expects = [(8, 20.3), (13, 32.9), (32, 81.1)]
    for n, expect in expects:
        got = training_hours(n, 76)
        if abs(got - expect) > 0.1:
            print("  FAIL: %d 对 @76min = %.1f h，应约 %.1f h（是否漏了 ×2？）"
                  % (n, got, expect))
            ok = False
        else:
            print("  PASS: %d 对 = %d 次 × 76min = %.1f h（文档写 %.1f h）"
                  % (n, n * PAIR_TO_RUNS, got, expect))
    # 反向验证：不乘 2 会得到 40.5，必须与 81.1 明显不同
    wrong = 32 * 76 / 60.0
    if abs(wrong - 81.1) < 1:
        print("  FAIL: 不乘 PAIR_TO_RUNS 也得到 81h，说明自检失效")
        ok = False
    else:
        print("  PASS: 漏乘时得 %.1f h ≠ 81.1 h → 自检能识别该错误" % wrong)

    print()
    print("*" * 76)
    print("教训（本项目真实发生过）：核对数字时，先确认**单位的定义**，再做乘法。")
    print("  「32 对」≠「32 次」——配对设计里一对含 2 次训练。")
    print("  看到一个看似矛盾的算式（76×32=40.5≠81）时，先问「32 的单位是什么」，")
    print("  而不是直接断定原数错了。**纠正一个不存在的错误，比原错误更危险**，")
    print("  因为它披着「我自查过了」的外衣。")
    print("*" * 76)

    print()
    print("总体：%s" % ("功率分析数字全部复核一致" if ok else "存在不一致"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
