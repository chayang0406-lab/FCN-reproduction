#!/usr/bin/env python3
"""E0 长任务守护：跑完自动汇总 + 失败自动诊断。

## 为什么需要这个
E0 要连续跑 8 个种子、每个约 76 分钟，总共约 10-11 小时。
这期间最怕两件事：
1. 某个种子跑到一半崩了，后面 7 个全白等（或更糟：脚本静默继续，结果一堆 0）
2. 跑完了但没人看结果，第二天才发现参数量全错

本脚本做两件事：
- **跑之前**：预检（数据存在？feature_map 对？参数量对？）
- **跑之后**：汇总 8 个种子的结果，算出 σ、CI、与基准的偏差，并给出
  「本次实验能分辨多大效应」（功率分析），直接产出报告可用的一段话

## 用法
  # 预检（几秒，建议先跑，确认没问题再开长任务）
  python3 e0_guard.py --workdir <runs> --data-root <data> --precheck
  # 运行后汇总
  python3 e0_guard.py --workdir <runs> --data-root <data> --summarize
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

AUTHOR_BASELINE = {
    "source": "FCNv2_Criteo_016_cde6d2ff.log (exp=4/lin=1, HEAD 默认配置)",
    "valid_auc": 0.815627,
    "test_auc": 0.816145,
    "n_params": 16532562,
}
EXPECT_PARAMS = 16532562
EXPECT_VOCAB_SUM = 910747


def parse_run_log(path):
    """从 fuxictr 日志分段解析。严格区分 valid / test。"""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    m = re.search(r"Total number of parameters: (\d+)", txt)
    out["n_params"] = int(m.group(1)) if m else None

    mv = re.search(r"Validation evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    mt = re.search(r"Test evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)", txt, re.S)
    out["valid_auc"] = float(mv.group(2)) if mv else None
    out["valid_logloss"] = float(mv.group(1)) if mv else None
    out["test_auc"] = float(mt.group(2)) if mt else None
    out["test_logloss"] = float(mt.group(1)) if mt else None

    # epoch 计时：fuxictr 只在第一个 epoch 打 start，后续只打 end
    from datetime import datetime
    # 【坑】不要用 ^ 锚行首：我们的 train.log 并入了 tqdm 进度条，
    # 时间戳可能不在行首（进度条在同一行内刷新），锚 ^ 会匹配不到，
    # 导致 ep=0 / s/ep=- 的静默错误（不影响 mean/sd，但会让人以为没测到）。
    # runner 实测发现，已去掉锚点。
    st = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Epoch=(\d+) start", txt)
    en = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Epoch=(\d+) end", txt)
    st = {int(e): datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f") for t, e in st}
    en = {int(e): datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f") for t, e in en}
    secs, prev = [], st.get(1)
    for i in sorted(en):
        if prev is not None and en[i] >= prev:
            secs.append((en[i] - prev).total_seconds())
        prev = en[i]
    out["n_epochs"] = len(en)
    out["mean_epoch_seconds"] = (sum(secs) / len(secs)) if secs else None
    out["epoch_seconds"] = secs
    return out


def precheck(args):
    print("=" * 66)
    print("E0 预检")
    print("=" * 66)
    ok = True

    # 1. 数据
    h5dir = os.path.join(args.data_root, "Criteo_x4_10_h5")
    for split in ("train", "valid", "test"):
        p = os.path.join(h5dir, "%s.h5" % split)
        if os.path.exists(p):
            print("  [OK]   %s.h5  %.1f GB" % (split, os.path.getsize(p) / 1e9))
        else:
            print("  [FAIL] 缺 %s.h5" % split)
            ok = False

    fmj = os.path.join(h5dir, "feature_map.json")
    if os.path.exists(fmj):
        fm = json.load(open(fmj))
        nonlocal_vocab = 0
        if isinstance(fm.get("features"), list):
            for item in fm["features"]:
                for k, v in item.items():
                    nonlocal_vocab += v.get("vocab_size", 0)
        elif isinstance(fm.get("features"), dict):
            for k, v in fm["features"].items():
                nonlocal_vocab += v.get("vocab_size", 0)
        print("  Σ(vocab_size) = %d (期望 %d) -> %s"
              % (nonlocal_vocab, EXPECT_VOCAB_SUM,
                 "OK" if nonlocal_vocab == EXPECT_VOCAB_SUM else "!! FAIL"))
        if nonlocal_vocab != EXPECT_VOCAB_SUM:
            ok = False
    else:
        print("  [FAIL] 缺 feature_map.json")
        ok = False

    # 2. 种子列表
    # 【坑，runner 实测发现】driver 写在 <workdir>/runs/seed_list.json，
    # 而这里原来读 <workdir>/seed_list.json，两处不一致 → precheck 永远打印
    # 「还没生成 seed_list.json（首次运行会生成）」，跑完 8 个种子后仍是这句，
    # 人会误以为还没生成而不去核对 → E3 手写种子 → 配对设计静默失效。
    # 已对齐到 <workdir>/runs/seed_list.json（与 driver 一致）。
    sl = os.path.join(args.workdir, "runs", "seed_list.json")
    if os.path.exists(sl):
        d = json.load(open(sl))
        print("  [OK]   seed_list.json: %s（E3 必须复用）" % d.get("seeds"))
        print("         路径: %s" % sl)
    else:
        print("  [FAIL] 缺 seed_list.json（找的是 %s）" % sl)
        print("         它是 E0 跑第一个种子时生成的；若已跑过，说明路径不对，需排查")
        ok = False

    # 3. 磁盘
    st = os.statvfs(args.data_root) if hasattr(os, "statvfs") else None
    if st:
        free_gb = st.f_bavail * st.f_frsize / 1e9
        print("  可用磁盘 %.0f GB" % free_gb)

    print("=" * 66)
    print("预检结果: %s" % ("全部通过，可以开跑" if ok else "有 FAIL，先修再跑"))
    return 0 if ok else 1


def summarize(args):
    runs_dir = os.path.join(args.workdir, "runs", "e0_baseline")
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(d):
            continue
        m = re.search(r"seed(\d+)", os.path.basename(d))
        log = os.path.join(d, "train.log")
        r = parse_run_log(log)
        if r is None:
            continue
        r["seed"] = int(m.group(1)) if m else None
        r["run_id"] = os.path.basename(d)
        r["param_ok"] = (r["n_params"] == EXPECT_PARAMS)
        r["has_result"] = (r["valid_auc"] is not None and r["test_auc"] is not None)
        # 判定该 run 是否可参与统计
        r["usable"] = bool(r["param_ok"] and r["has_result"])
        if not r["param_ok"]:
            r["exclude_reason"] = ("参数量 %s ≠ %d" % (r["n_params"], EXPECT_PARAMS)
                                   if r["n_params"] is not None else "解析不到参数量")
        elif not r["has_result"]:
            r["exclude_reason"] = "日志里没有 valid/test 结果（可能崩溃）"
        else:
            r["exclude_reason"] = None
        rows.append(r)

    if not rows:
        print("没找到任何 run 日志（找的是 %s）" % runs_dir)
        return 1

    # 【关键修正】只有 usable 的 run 才进统计。
    # 之前的版本把参数量错误 / 崩溃的 run 也算进 mean/sd，
    # 会把功率分析严重放大（实测 δ_min 被放大 3×，需要配对数被放大 8.9×）。
    usable = [r for r in rows if r["usable"]]
    excluded = [r for r in rows if not r["usable"]]

    va = [r["valid_auc"] for r in usable]
    ta = [r["test_auc"] for r in usable]

    print("=" * 92)
    print("E0 汇总：共发现 %d 个 run，其中可用 %d 个，排除 %d 个"
          % (len(rows), len(usable), len(excluded)))
    print("=" * 92)
    print("%-6s %-12s %-12s %-12s %-4s %-9s %-12s %s"
          % ("seed", "valid_auc", "test_auc", "valid_ll", "ep", "s/ep", "params", "status"))
    for r in rows:
        status = "OK" if r["usable"] else ("EXCLUDED: " + (r["exclude_reason"] or ""))
        print("%-6s %-12s %-12s %-12s %-4s %-9s %-12s %s"
              % (r["seed"],
                 "%.6f" % r["valid_auc"] if r["valid_auc"] is not None else "-",
                 "%.6f" % r["test_auc"] if r["test_auc"] is not None else "-",
                 "%.6f" % r["valid_logloss"] if r["valid_logloss"] is not None else "-",
                 r["n_epochs"],
                 "%.1f" % r["mean_epoch_seconds"] if r["mean_epoch_seconds"] else "-",
                 r["n_params"] if r["n_params"] is not None else "-",
                 status))

    if excluded:
        print("\n!! 已排除 %d 个 run（不参与任何统计）：" % len(excluded))
        for r in excluded:
            print("   - %s : %s" % (r["run_id"], r["exclude_reason"]))
        print("   这些 run 的结果必须作废，报告里不得引用。")

    if len(va) >= 2:
        mean_v = statistics.mean(va)
        sd_v = statistics.stdev(va)
        mean_t = statistics.mean(ta) if ta else None
        sd_t = statistics.stdev(ta) if len(ta) >= 2 else None
        print()
        print("valid AUC : mean %.6f  sd %.6f  [%.6f, %.6f]"
              % (mean_v, sd_v, min(va), max(va)))
        if mean_t:
            print("test  AUC : mean %.6f  sd %.6f" % (mean_t, sd_t or 0))
        print("作者基准  : valid %.6f / test %.6f  (%s)"
              % (AUTHOR_BASELINE["valid_auc"], AUTHOR_BASELINE["test_auc"],
                 AUTHOR_BASELINE["source"]))
        dv = mean_v - AUTHOR_BASELINE["valid_auc"]
        print("valid 偏差: %+.6f  (%.2f 个 σ)" % (dv, dv / sd_v if sd_v else 0))
        if mean_t:
            dt = mean_t - AUTHOR_BASELINE["test_auc"]
            print("test  偏差: %+.6f" % dt)
            print("  >> 注意 valid 与 test 偏差符号是否一致；不一致时不能得出单一结论")

        # 功率分析
        print()
        print("-" * 78)
        print("功率分析（α=0.05, power=0.80, 配对设计）")
        print("  公式 n = 7.849 × σ² / δ²")
        print("%-14s %-10s %s" % ("要分辨 δ", "需要配对数", "折算训练次数/小时(按76分钟)"))
        for delta in (0.0005, 0.0008, 0.001, 0.002):
            n = int(7.849 * sd_v ** 2 / delta ** 2 + 0.999)
            n = max(n, 1)
            print("%-14.4f %-10d %d 次 / %.1f 小时"
                  % (delta, n, 2 * n, 2 * n * 76 / 60))
        dmin = 2.8016 * sd_v / (len(va) ** 0.5)
        print()
        print("本次实验分辨率：n=%d, σ=%.6f  =>  δ_min = %.6f"
              % (len(va), sd_v, dmin))
        print("  >> 可直接写进报告：")
        print("     「本次实验用 %d 个随机种子。以实测 σ=%.6f 计，在 α=0.05、power=0.80 下，"
              % (len(va), sd_v))
        print("       本实验能分辨的最小效应为 δ_min = %.6f。因此 ΔAUC < %.6f 的差异"
              % (dmin, dmin))
        print("       本实验无法分辨，不据此宣称改进有效。」")

        out = {
            "n_runs_found": len(rows),
            "n_valid": len(usable),
            "n_excluded": len(excluded),
            "excluded_runs": [{"run_id": r["run_id"], "reason": r["exclude_reason"]}
                              for r in excluded],
            "n_seeds": len(va), "valid_mean": mean_v, "valid_sd": sd_v,
            "valid_min": min(va), "valid_max": max(va),
            "test_mean": mean_t, "test_sd": sd_t,
            "author_baseline": AUTHOR_BASELINE,
            "valid_delta_vs_author": dv,
            "delta_min_resolvable": dmin,
            "runs": rows,
        }
        p = os.path.join(args.workdir, "runs", "e0_final_summary.json")
        json.dump(out, open(p, "w"), indent=2)
        print("\n汇总已写入 %s" % p)
    else:
        print("\n有效 run 不足 2 个，无法统计")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        sys.exit(precheck(args))
    elif args.summarize:
        sys.exit(summarize(args))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

