"""FCNv2 参数量校验工具 —— 给 runner 用，每次实验后独立核对参数量。

⚠️ 关于「旧版公式」的重要更正（2026-11-18，infra）
--------------------------------------------------------------------
上一轮我曾给出「旧版每层 = w(D×D/3) + b(D) = 130,416，6 层 → 16,138,194，残差 0」。
**这个推导是错的。** 它只在 ECN=3/LCN=3（L=6）这一个点上成立，是单点巧合：

    L      D//3 公式     实际拟合律     一致?
    4      15,616,530    16,038,354    NO
    5      15,877,362    16,088,274    NO
    6      16,138,194    16,138,194    YES  <-- 唯一相交点
    7      16,399,026    16,188,114    NO
    8      16,659,858    16,238,034    NO

斜率对比更直接暴露问题：D//3 公式隐含每层增量 2×130,416 = 260,832，
而 14 份旧版日志的实际拟合斜率只有 **49,920**。差 5 倍多。

真相：把 Criteo 目录下 **全部 14 份旧版日志**做线性拟合，得到
    total = 15,838,674 + 49,920 × (ECN_layers + LCN_layers)
该式对 14/14 份日志**逐份精确命中（残差全 0）**，L 覆盖 4–10。
但这只是**经验线性律**，不是结构推导——旧版源码不在本仓库
（旧日志里 `Load best model` 路径是 `/root/autodl-tmp/model_zoo/DCNv3/DCNv3_torch/`，
说明这些日志来自另一个项目 DCNv3_torch，非本 FCN 仓库），
且 `git log --all -- src/FCNv2.py` 只有 1 次提交，仓库从未包含旧版实现。
因此旧版的真实层次结构**无法确定**，只能给经验拟合式。

结论：本工具对**新版（HEAD）**给结构推导的闭式公式（已 27/27 命中作者日志）；
对**旧版**只给经验拟合式，并明确标注「结构未确定」。
"""


# ============================================================
# 新版（HEAD / KDD'26 分支）—— 结构推导，闭式公式
# ============================================================
def count_params(exp_num_layers,
                 lin_num_layers,
                 num_fields=39,
                 embedding_dim=16,
                 vocab_total=910747,
                 num_heads=1,
                 batch_norm=False,
                 layer_norm=False):
    """HEAD 版 FCNv2 参数量（结构推导）。

    依据 src/FCNv2.py（KDD'26 分支 HEAD）：
      embedding: 每特征一张 vocab_size × (embedding_dim×num_heads) 表
      D = num_fields × embedding_dim × num_heads
      两个交叉网络（E2LCN + L2ECN），各 (exp+lin) 层，每层：
          w     = Linear(D, D//2, bias=False) -> D×(D//2)
          gamma = Parameter(D//2)             -> D//2      (layer_norm=False)
          beta  = Parameter(D//2)             -> D//2      (layer_norm=False)
          b     = Parameter(D)                -> D
      每网络末尾 fc = Linear(D,1) -> D+1

    默认参数对应 Criteo_x4（39 字段 / dim16 / vocab 合计 910,747 / heads1）。
    """
    D = num_fields * embedding_dim * num_heads
    emb = vocab_total * embedding_dim * num_heads

    per_layer = D * (D // 2) + D                 # w + b
    if layer_norm:
        per_layer += 2 * (D // 2)                # LayerNorm weight+bias
    else:
        per_layer += (D // 2) + (D // 2)         # gamma + beta
    if batch_norm:
        per_layer += 2 * num_heads               # BatchNorm1d(num_heads)

    n_layers = exp_num_layers + lin_num_layers
    per_net = n_layers * per_layer + (D + 1)
    return emb + 2 * per_net


def count_params_old_impl(ecn_layers, lcn_layers, **_ignored):
    """旧版参数量 —— **经验拟合式，非结构推导**（结构未确定，见模块开头说明）。

    total = 15,838,674 + 49,920 × (ecn_layers + lcn_layers)

    来源：对 checkpoints/Criteo_x4_10_h5/ 下 14 份旧版日志（L 覆盖 4–10）
    做线性拟合，14/14 精确命中，残差全 0。
    仅在 Criteo_x4（emb_dim=16, vocab 合计 910,747）标定，**外推到其他数据集不可靠**。

    验证点：ecn_layers=3, lcn_layers=3 -> 16,138,194（作者主日志）
    """
    return 15838674 + 49920 * (ecn_layers + lcn_layers)


# 与真实模型对照的一行式工具（runner 直接调用）
def check_model_params(model, expect_n_params, label=""):
    """拿一个已建好的 torch 模型，核对其参数量是否等于期望值。

    返回 (ok, actual, expect, msg)。不抛异常，方便 runner 记录后继续。
    """
    actual = sum(p.numel() for p in model.parameters())
    ok = actual == expect_n_params
    if ok:
        msg = "OK  %s 参数量 = %d（与配置相符）" % (label, actual)
    else:
        msg = ("MISMATCH %s 实际 = %d，期望 = %d，差 %+d\n"
               "  >>> 配置未生效 / 代码被改动 / feature_map 不匹配，请检查！"
               % (label, actual, expect_n_params, actual - expect_n_params))
    return ok, actual, expect_n_params, msg


# ============================ 单元测试 ============================
def _check(label, got, want):
    ok = got == want
    print("  %-56s got=%-10s want=%-10s %s" % (label, got, want, "OK" if ok else "FAIL"))
    return ok


def selftest():
    all_ok = True

    print("=" * 96)
    print("A. 新版闭式公式 vs 作者日志实测参数量（结构推导）")
    print("=" * 96)
    all_ok &= _check("exp=4, lin=1  (HEAD 默认, = _016 日志)", count_params(4, 1), 16532562)
    all_ok &= _check("exp=4, lin=2  (= _017 日志)", count_params(4, 2), 16924434)
    all_ok &= _check("exp=4, lin=5  (= _020 / _024 日志)", count_params(4, 5), 18100050)

    print()
    print("=" * 96)
    print("B. 旧版经验拟合式 vs 作者主日志")
    print("=" * 96)
    all_ok &= _check("ECN=3, LCN=3 (= FCNv2_Criteo.log 主日志)",
                     count_params_old_impl(3, 3), 16138194)

    print()
    print("=" * 96)
    print("C. 新版公式 —— 作者日志全量交叉验证")
    print("=" * 96)
    for e, l, want in [
        (1, 1, 15356946), (1, 2, 15748818), (2, 1, 15748818), (2, 2, 16140690),
        (1, 3, 16140690), (2, 3, 16532562), (3, 2, 16532562), (3, 3, 16924434),
        (5, 1, 16924434), (3, 4, 17316306), (4, 3, 17316306), (4, 4, 17708178),
        (5, 3, 17708178), (5, 4, 18100050), (5, 5, 18491922),
    ]:
        all_ok &= _check("exp=%d, lin=%d" % (e, l), count_params(e, l), want)

    print()
    print("=" * 96)
    print("D. 旧版经验式 —— 14 份日志全量验证（这才是正确性的硬证据）")
    print("=" * 96)
    for e, l, want in [
        (3, 1, 16038354), (3, 2, 16088274), (4, 1, 16088274), (3, 3, 16138194),
        (5, 1, 16138194), (3, 4, 16188114), (5, 2, 16188114), (3, 5, 16238034),
        (4, 4, 16238034), (5, 3, 16238034), (4, 5, 16287954), (5, 4, 16287954),
        (5, 5, 16337874),
    ]:
        all_ok &= _check("ECN=%d, LCN=%d" % (e, l), count_params_old_impl(e, l), want)

    print()
    print("=" * 96)
    print("E. 结构不变量（新版）")
    print("=" * 96)
    per_layer = 624 * 312 + 624 + 312 + 312
    all_ok &= _check("per_layer 常量", per_layer, 195936)
    all_ok &= _check("加一层的增量 == 2 × per_layer",
                     count_params(4, 2) - count_params(4, 1), 2 * per_layer)
    all_ok &= _check("20 层 = 15 层 + 5×2×per_layer",
                     count_params(4, 16), count_params(4, 1) + 15 * 2 * per_layer)

    print()
    print("=" * 96)
    print("结论：", "ALL PASS —— 公式可用于校验" if all_ok else "有 FAIL，公式有问题，勿用")
    print("=" * 96)
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)

