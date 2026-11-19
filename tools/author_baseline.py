#!/usr/bin/env python3
"""作者基准常量（单一来源）。所有工具报「与作者对比」时必须 import 这个。

=============================================================================
为什么要有这个文件
-----------------------------------------------------------------------------
2026-11-18 一晚之内，团队在不同地方重复犯了同一个错误：
**文档里写对了基准是 `_016`，实际做对比时却拿了主日志 `FCNv2_Criteo.log`。**

真实案例（captain 自查发现）：
    第一次对比：  我们 0.804883 vs 0.805413 → 差 -0.000530
                  0.805413 来自 FCNv2_Criteo.log = 旧实现（3ECN+3LCN, 16,138,194）← 错
    正确对比：    我们 0.804883 vs 0.804909 → 差 -0.000026
                  0.804909 来自 _016_cde6d2ff.log = HEAD（4+1, 16,532,562）← 对
**差 20 倍。** 若按错的写进报告，「复现精度」这一节就整个错了方向。

根因不是「不知道」，而是「知道了但执行时不去读文档」。
所以把基准写进**代码常量**，所有工具从这里取，并在输出里**打印基准日志文件名**——
这样一旦用错，输出里能直接看出来。

=============================================================================
关于 epochs 的口径（容易搞错，特别说明）
-----------------------------------------------------------------------------
`_016` 日志里：
    "Epoch=N end" 行数     = 18
    "Epoch=N start" 行数   = 1     （fuxictr 只打一次）
    "Epoch==19 early stop" = 第 19 轮早停

=> **实际训练轮数 = 19**（第 19 轮跑完评估后触发早停，只打 early stop 不打 end）

所以：
    epochs_trained   = 19   ← 报告里写这个
    epochs_end_lines = 18   ← 解析日志时 end 行只有 18 个，别把它当训练轮数

早期文档里写过「19 epoch」和「20 epoch」，也写过「18」——都是指不同东西：
    19 = 实际训练轮数（正确）
    18 = end 行数（解析中间量）
    20 = 误记（主日志 FCNv2_Criteo.log 才是 20 轮，那是旧实现）

=============================================================================
"""
from __future__ import absolute_import, division

# ---------------------------------------------------------------------------
# E0 的对照基准：HEAD 代码 + exp=4/lin=1 + dim=16（与数据集无关的通用量）
# 出处：code/FCN_repo/checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_016_cde6d2ff.log
# ---------------------------------------------------------------------------
AUTHOR_BASELINE = {
    "log": "FCNv2_Criteo_016_cde6d2ff.log",
    "log_path_hint": "code/FCN_repo/checkpoints/Criteo_x4_10_h5/",
    "config": "exp=4/lin=1, embedding_dim=16, batch_size=8192, lr=1e-3",
    "n_params": 16532562,
    "valid_auc": 0.815627,
    "valid_logloss": 0.436533,
    "test_auc": 0.816145,
    "test_logloss": 0.436048,
    # 实际训练轮数（= end 行 18 + 第 19 轮 early stop）
    "epochs_trained": 19,
    "epochs_end_lines": 18,
    "early_stop_epoch": 19,
    # 单轮耗时：不含 eval 的纯训练 / 含 eval 的总计
    "mean_epoch_seconds": 204.38,       # 含 valid 评估（_016 实测）
    "epoch_seconds_range": (204.03, 205.12),
    "hw": "RTX 4090",
}

# ---------------------------------------------------------------------------
# 明确标注「不要用这个当 E0 基准」
# ---------------------------------------------------------------------------
NOT_A_BASELINE = {
    "FCNv2_Criteo.log": {
        "why": "旧实现（Params 用 ECN_num_layers/LCN_num_layers，3ECN+3LCN），"
               "用当前 HEAD 代码在任何 exp/lin 组合下都复现不出 16,138,194",
        "n_params": 16138194,
        "valid_auc": 0.815699,
        "test_auc": 0.816099,
        "epochs_trained": 20,
        "mean_epoch_seconds": 191.22,
        "use_for": "仅作历史参考 / 说明「作者改过实现」",
    },
    "FCNv2_Criteo_017_30a838c6.log": {
        "why": "配置是 exp=4/lin=2（16,924,434），与 E0 的 4/1 不同配置",
        "n_params": 16924434,
        "valid_auc": 0.815818,
        "test_auc": 0.816352,
        "epochs_trained": 21,
        "mean_epoch_seconds": 219.73,
        "use_for": "仅作「配置敏感性」参考，不能与 E0 比性能",
    },
}


def format_baseline_line(prefix="基准"):
    """给工具输出用的单行基准说明。带文件名，用错基准时一眼能看出来。"""
    b = AUTHOR_BASELINE
    return ("%s：%s（%s，%s 参数）\n"
            "        valid %.6f / %.6f   test %.6f / %.6f\n"
            "        %d 轮训练，%.2f s/epoch（%s）"
            % (prefix, b["log"], b["config"], "{:,}".format(b["n_params"]),
               b["valid_auc"], b["valid_logloss"], b["test_auc"], b["test_logloss"],
               b["epochs_trained"], b["mean_epoch_seconds"], b["hw"]))


def compare(valid_auc=None, test_auc=None, n_params=None, mean_epoch_seconds=None):
    """与基准逐项对比，返回对比表（含「差几个 σ」需要外部 σ）。"""
    b = AUTHOR_BASELINE
    rows = []
    if valid_auc is not None:
        rows.append(("valid AUC", b["valid_auc"], valid_auc,
                     valid_auc - b["valid_auc"]))
    if test_auc is not None:
        rows.append(("test AUC", b["test_auc"], test_auc, test_auc - b["test_auc"]))
    if n_params is not None:
        rows.append(("参数量", b["n_params"], n_params, n_params - b["n_params"]))
    if mean_epoch_seconds is not None:
        rows.append(("s/epoch", b["mean_epoch_seconds"], mean_epoch_seconds,
                     mean_epoch_seconds - b["mean_epoch_seconds"]))
    return rows


if __name__ == "__main__":
    print(format_baseline_line())
    print()
    print("不要用作 E0 基准的日志：")
    for name, d in NOT_A_BASELINE.items():
        print("  %s" % name)
        print("    原因：%s" % d["why"])
        print("    %s 个参数 / valid %.6f / %d 轮 / %.2f s/epoch"
              % ("{:,}".format(d["n_params"]), d["valid_auc"],
                 d["epochs_trained"], d["mean_epoch_seconds"]))
        print("    仅用于：%s" % d["use_for"])
