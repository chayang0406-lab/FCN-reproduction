"""从作者 FCNv2 日志里，分别提取 valid AUC 和 test AUC。

【修正说明 —— 2026-11-18】
前一版脚本有个**系统性错误**：它把所有 `[Metrics] AUC: x` 取最大值当作 "best_auc"。
但日志里 AUC 出现在三个位置：
  1. 每个 epoch 结束后的验证集评估   -> 这是 valid AUC
  2. "****** Validation evaluation ******" 段（训练结束后用 best model 再评一次）-> valid AUC
  3. "******** Test evaluation ********" 段 -> 这是 test AUC
而 test AUC 通常比 valid AUC 略高（因为早停是看着 valid 挑的，
best model 在 valid 上被"挑过"，在 test 上是干净的）。
所以「取全局最大值」几乎总是取到 test AUC，
把它标成 valid AUC 会得出完全错误的对照结论。

现在改为显式分段解析：分别给出
  - valid_epoch_best : 逐 epoch 验证 AUC 的最大值（早停实际依据）
  - valid_final      : 训练结束后 Validation evaluation 段的 AUC
  - test             : Test evaluation 段的 AUC
"""
import os
import re
import glob


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    d = {"file": os.path.basename(path)}

    # 参数量
    m = re.search(r"Total number of parameters: (\d+)", "".join(lines))
    d["n_params"] = int(m.group(1)) if m else None

    # 层数参数名体系
    text = "".join(lines)
    m = re.search(r'"(exp_num_layers)": "(\d+)"', text)
    if m:
        d["naming"] = "exp/lin(HEAD)"
        d["exp"] = int(m.group(2))
        d["lin"] = int(re.search(r'"(lin_num_layers)": "(\d+)"', text).group(2))
    else:
        m = re.search(r'"(ECN_num_layers)": "(\d+)"', text)
        if m:
            d["naming"] = "ECN/LCN(old)"
            d["exp"] = int(m.group(2))
            d["lin"] = int(re.search(r'"(LCN_num_layers)": "(\d+)"', text).group(2))
        else:
            d["naming"], d["exp"], d["lin"] = "none", None, None

    # 分段：找到 Validation evaluation / Test evaluation 的行号
    idx_valid_final = None
    idx_test = None
    epoch_valid = []
    for i, ln in enumerate(lines):
        if "Validation evaluation" in ln:
            idx_valid_final = i
        elif "Test evaluation" in ln:
            idx_test = i
        elif "[Metrics]" in ln and idx_valid_final is None:
            # 出现在 Validation evaluation 之前的 [Metrics] = 逐 epoch 验证
            m2 = re.search(r"AUC: (0\.\d+)", ln)
            if m2:
                epoch_valid.append(float(m2.group(1)))

    def metric_after(idx, key="AUC"):
        if idx is None:
            return None
        for j in range(idx + 1, min(idx + 12, len(lines))):
            m2 = re.search(key + r": (0\.\d+)", lines[j])
            if m2:
                return float(m2.group(1))
        return None

    d["valid_epoch_best"] = max(epoch_valid) if epoch_valid else None
    d["valid_final"] = metric_after(idx_valid_final)
    d["test_auc"] = metric_after(idx_test)
    d["valid_final_ll"] = metric_after(idx_valid_final, "logloss")
    d["test_ll"] = metric_after(idx_test, "logloss")
    d["epochs_run"] = len(epoch_valid)
    return d


def main():
    # 脚本全目录扫描（Windows 路径写死为作者仓库位置）
    LOG_DIR = r"D:\中南大学\数据挖掘\code\FCN\checkpoints\Criteo_x4_10_h5"
    files = sorted(glob.glob(os.path.join(LOG_DIR, "FCNv2_*.log")))
    rows = [parse(p) for p in files]
    rows = [r for r in rows if r["n_params"]]

    hdr = "%-42s %-15s %3s %3s %10s %10s %10s %9s %9s" % (
        "file", "naming", "exp", "lin", "n_params", "valid_best", "valid_final", "test", "ep")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["n_params"] or 0):
        print("%-42s %-15s %3s %3s %10s %10s %10s %9s %9s" % (
            r["file"], r["naming"], r["exp"], r["lin"], r["n_params"],
            r["valid_epoch_best"], r["valid_final"], r["test_auc"], r["epochs_run"]))

    print()
    print("=" * 100)
    print("【E0 对照基准】(HEAD 代码, exp=4/lin=1, 即 config/model_config.yaml 的 FCNv2_Criteo)")
    for r in rows:
        if r["file"] == "FCNv2_Criteo_016_cde6d2ff.log":
            print("  文件        : %s" % r["file"])
            print("  参数名体系  : %s (exp=%d, lin=%d)" % (r["naming"], r["exp"], r["lin"]))
            print("  参数量      : %d" % r["n_params"])
            print("  valid AUC   : %.6f  (logloss %.6f)" % (r["valid_epoch_best"], r["valid_final_ll"]))
            print("  test  AUC   : %.6f  (logloss %.6f)" % (r["test_auc"], r["test_ll"]))
            print("  训练轮数    : %d" % r["epochs_run"])

    print()
    print("【对照：HEAD 默认配置下所有日志的 valid / test 分列】")
    head41 = [r for r in rows if r["naming"].startswith("exp") and r["exp"] == 4 and r["lin"] == 1]
    for r in head41:
        print("  %-42s valid=%.6f  test=%s" % (
            r["file"], r["valid_epoch_best"],
            ("%.6f" % r["test_auc"]) if r["test_auc"] else "n/a"))

    print()
    print("【valid 最高 vs test 最高（注意两者可能不是同一份日志！）】")
    withv = [r for r in rows if r["valid_epoch_best"]]
    print("  valid AUC 最高: %-42s %.6f (exp=%s lin=%s)" % tuple(
        [x for x in [(lambda r: (r["file"], r["valid_epoch_best"], r["exp"], r["lin"]))(
            max(withv, key=lambda x: x["valid_epoch_best"]))]] [0]))
    witht = [r for r in rows if r["test_auc"]]
    mt = max(witht, key=lambda x: x["test_auc"])
    print("  test  AUC 最高: %-42s %.6f (exp=%s lin=%s)" % (mt["file"], mt["test_auc"], mt["exp"], mt["lin"]))
    print()
    print("  主日志 FCNv2_Criteo.log:")
    for r in rows:
        if r["file"] == "FCNv2_Criteo.log":
            print("    valid_epoch_best = %s, valid_final = %s, test = %s"
                  % (r["valid_epoch_best"], r["valid_final"], r["test_auc"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

