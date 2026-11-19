import numpy as np

print("=" * 78)
print("BSW-Tri 设计的数学分析：为什么不能同时满足四个条件")
print("=" * 78)
print("""
作者权重：w = max(0, Δ)，Δ = L_aux - L_main

我们希望同时满足：
  C1  w(Δ≤0) = 0        与作者一致：学得好的腿不奖不罚
  C2  w 单调递增         差得越多权重越大
  C3  w 有上界 alpha     不淹没主损失（作者缺失）
  C4  w 处处可导          不产生死梯度（作者缺失）

【定理】C1 + C2 + C3 + C4 互不相容。
证明：设 f 满足 C1/C2/C4 且 f ≥ 0。
  C1 给出 f(x) = 0 对所有 x ≤ 0。
  C2（递增）给出 f(x) ≥ f(0) = 0 对 x > 0，故 f ≥ 0 全局成立（已由 C1 覆盖）。
  由 C1，f 在 x < 0 上恒为 0，故左导数 f'(0⁻) = 0。
  C4 要求 f'(0⁻) = f'(0⁺)。若 f 在 x>0 严格递增，则 f'(0⁺) > 0 ≠ 0，矛盾。
  ∴ 必须放弃 C1 或 C4（或让 f 在 0 右侧也平坦，即恒为 0，退化为无改进）。
""")

print("=" * 78)
print("三种取舍，各自放弃什么")
print("=" * 78)
print("""
取舍甲：保留 C1，放弃 C4（作者的选择 + 加有界）
    w = alpha · tanh( beta · relu(Δ) / L )
    - Δ≤0 : w = 0                        ✅ 与作者完全一致
    - Δ>0 : w = alpha·tanh(beta·Δ/L)     ✅ 有界、尺度归一
    - Δ=0 : 一阶不可导（但与作者同为单点，测度零）
    新增好处：给作者的无界权重加上限，且用 L 归一化尺度

取舍乙：保留 C4，放弃 C1（允许负权重）
    w = alpha · tanh( beta · Δ / L )
    - Δ<0 时 w < 0 → 优化器会抬高该腿损失  ❌ 语义错误，否决

取舍丙：保留 C1，C4 改为"几乎处处可导"（softplus 平滑 relu）
    w = alpha · tanh( beta · (softplus(Δ) - ln2) / L )
    - Δ=0 时恰为 0  ✅
    - Δ<0 时 w < 0   ❌ 又回到乙的问题
""")

print("=" * 78)
print("★ 决定采用「取舍甲」：w = alpha · tanh( beta · relu(Δ) / L )")
print("=" * 78)


def author_w(d):
    return max(0.0, d)


def bsw(d, alpha=1.0, beta=1.0, L=0.45):
    x = beta * max(0.0, d) / L
    return alpha * np.tanh(x)


print("%-10s %14s %14s %12s" % ("Δ", "作者 max(0,Δ)", "BSW-Tri", "二者比值"))
for d in [-0.5, -0.1, -0.01, 0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
    a = author_w(d)
    b = bsw(d)
    ratio = ("%.3f" % (b / a)) if a > 1e-12 else "—"
    print("%-10.3f %14.6f %14.6f %12s" % (d, a, b, ratio))

print()
print("关键差异（这就是可写进报告的效果）：")
for d in [0.5, 1.0, 5.0, 20.0]:
    a = author_w(d)
    b = bsw(d)
    print("  Δ=%6.1f : 作者 w=%8.4f → 总量级 %-8.2f ; BSW w=%.6f → 封顶 %.6f"
          % (d, a, a, b, 1.0))
print()
print("  作者的 w 随 Δ 线性增长（无上界）；BSW 在 Δ/L ≳ 2 后饱和到 alpha。")
print("  早期训练 L_aux 远大于 L 时，作者的辅助项会主导总损失，BSW 不会。")

print()
print("=" * 78)
print("尺度归一化：为什么除以 L 重要")
print("=" * 78)
print("""
同一 Δ 在不同数据集/训练阶段的含义不同：
  - 训练初期 L≈0.69（随机初始化），Δ=0.1 是 14% 的相对差
  - 训练后期 L≈0.44，Δ=0.1 是 23% 的相对差
除以 L 后，权重反映的是"相对差距"，跨数据集可比。
作者的绝对 Δ 在 L 尺度不同的数据集上行为不一致。
""")
for L in [0.69, 0.44]:
    print("  L=%.2f, Δ=0.1 : 作者 w=%.4f ; BSW w=%.6f"
          % (L, author_w(0.1), bsw(0.1, L=L)))
