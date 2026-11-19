#!/usr/bin/env python3
"""E0 基线复现 + 种子噪声实测（在 DCU 上跑真实 Criteo 数据）。

=============================================================================
本文件 = tools/run_e0_baseline.py 的**最小补丁版**（5 处 hunks，不重写实验逻辑）。
原文件保留不动；逐条差异见文件末尾「PATCH LOG」。
为什么必须补：原版在真实环境上**第一步就会崩**，且崩溃后会被静默跳过、伪造成
`param_check: OK` 的 null 结果（下面每条都有实测证据）。
=============================================================================

设计要点（为什么这么写）：
1. **配对设计的基础**：所有种子写成列表落盘，E3 改进实验必须用同一份列表，
   这样种子效应可相消，把需要的样本量从「独立 16」降到「配对 8」。
2. **每个 run 独立目录**：不覆盖，便于任何一次结果都能追溯。
3. **参数量自动校验**：每个 run 都用闭式公式核对实际参数量，对不上立刻**中止**，
   防止「配置没生效却不知道」，也防止白跑 76 分钟 × N。
4. **valid / test 严格分列**：早停是在 valid 上选的模型，所以 test 系统性偏高，
   两者混列会得出反向结论（这一点已被实际日志验证过）。

用法：
  python3 run_e0_baseline_patched.py --repo <FCN> --workdir <dir> --data-root <dir> --seeds 2025,2026,...
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

DATASET_KEY = "Criteo_x4_10_h5"
# 仓库 dataset_config.yaml 里唯一存在的 Criteo 条目，借它的 feature_cols / label_col
DATASET_KEY_FALLBACK = "Criteo_x4_001"

# ---------------------------------------------------------------------------
# fuxictr 2.0.1 兼容垫片
# ---------------------------------------------------------------------------
# 事实（2026-11-18 实测）：仓库 run_expid.py 第 24 行
#     from fuxictr.utils import ... , delete_model_files
# 但 PyPI 上 fuxictr-2.0.1 wheel 的 fuxictr/utils.py **没有** delete_model_files
# （整个 fuxictr 包里都搜不到这个名字）。
# => 原样运行必然 ImportError，每个 run 秒退。
#
# 解法：不改仓库任何文件。每次 run 生成一个启动器，在 import run_expid 之前把该
# 符号补进 fuxictr.utils 命名空间，再用 runpy 执行 run_expid.py。
# 该垫片只做「删除 model_dir 下以 model_id 开头的训练产物」，不参与任何数值计算，
# 因此**不影响模型行为与任何指标**。启动器每次 run 落盘并记 md5，可追溯可复现。
COMPAT_LAUNCHER = '''# -*- coding: utf-8 -*-
"""E0 run 启动器：给 fuxictr 2.0.1 补 delete_model_files，然后执行仓库 run_expid.py。

自动生成，请勿手改。来源见 tools/run_e0_baseline_patched.py 顶部说明。
"""
import os
import runpy
import sys

import fuxictr.utils as _u

if not hasattr(_u, "delete_model_files"):
    def delete_model_files(model_dir, model_id):
        """删除 model_dir 下文件名以 model_id 开头的文件（上游同名函数语义）。"""
        if not os.path.isdir(model_dir):
            return
        for filename in os.listdir(model_dir):
            if filename.startswith(model_id):
                fpath = os.path.join(model_dir, filename)
                if os.path.isfile(fpath):
                    os.remove(fpath)
    _u.delete_model_files = delete_model_files

_script = sys.argv[1]
sys.argv = sys.argv[1:]
# `python run_expid.py` 会把脚本所在目录放进 sys.path[0]；runpy 不会，手工补上，
# 否则 run_expid.py 里的 `import src as model_zoo` 会失败。
sys.path.insert(0, os.path.dirname(os.path.abspath(_script)))
runpy.run_path(_script, run_name="__main__")
'''


def param_formula(exp_layers, lin_layers, d=624, emb_dim=16, vocab_total=910747):
    """FCNv2 闭式参数量。与 tools/param_formula.py 同源，用于独立校验。"""
    per_layer = d * (d // 2) + (d // 2) + (d // 2) + d
    one_net = (exp_layers + lin_layers) * per_layer + (d + 1)
    return vocab_total * emb_dim + 2 * one_net


_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)"


def parse_log(path):
    """从 fuxictr 日志里分段解析，避免把 test 当 valid。"""
    txt = open(path, encoding="utf-8", errors="replace").read()

    m = re.search(r"Total number of parameters: (\d+)", txt)
    n_params = int(m.group(1)) if m else None

    # 逐 epoch 的验证 AUC
    epoch_auc = [float(a) for a, _ in
                 re.findall(r"Metrics\] AUC: ([\d.]+) - logloss: ([\d.]+)", txt)]

    # 训练结束后的两次评估：先 Validation，再 Test
    mv = re.search(r"Validation evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    mt = re.search(r"Test evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    valid = (float(mv.group(2)), float(mv.group(1))) if mv else (None, None)
    test = (float(mt.group(2)), float(mt.group(1))) if mt else (None, None)

    # [PATCH P4] 单轮耗时：fuxictr fit() **只在第一个 epoch 打 "Epoch=1 start"**，
    # 之后每轮只打 "Epoch=N end"。用 start 行差分 => 得到空列表、n_epochs 恒为 1、
    # mean_epoch_seconds 恒为 None，而且不报错（静默失效）。
    # 正确算法：
    #     epoch_1 = end_1 - start_1
    #     epoch_N = end_N - end_{N-1}     (N >= 2)
    #
    # [PATCH P4b] **不能**用 `^时间戳` 锚定行首。我们自己跑时把 stderr 也并进了
    # train.log（否则 traceback 会丢），而 tqdm 进度条写 stderr 且**不换行**，于是
    # 物理行会变成：
    #     100%|██████████| 1/1 [00:18<00:00, 18.58s/it]2026-11-18 21:38:58,552 ... Epoch=1 end
    # 用 `^TS` 就一条都匹配不到（实测 end_lines=0，静默返回空列表——正是 captain 说的
    # 那个坑的变种）。改为「时间戳出现在行内任意位置」，并允许中间夹杂任何非换行字符。
    #
    # 另外早停的那个 epoch 往往只打 "... early stop"，不打 "end"，所以 end 行数
    # 可能比实际轮数少 1（作者 19 轮日志实测 ends=18）。这一点在下游统计时要注意。
    starts = re.findall(_TS + r" .*?Epoch=(\d+) start ", txt)
    ends = re.findall(_TS + r" .*?Epoch=(\d+) end ", txt)
    epochs = []            # [(epoch_no, seconds)]
    prev_ts = None
    if starts:
        s_ts, s_no = starts[0]
        prev_ts = datetime.strptime(s_ts, "%Y-%m-%d %H:%M:%S,%f")
    for ts, no in ends:
        cur = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        if prev_ts is not None:
            epochs.append((int(no), (cur - prev_ts).total_seconds()))
        prev_ts = cur
    secs = [s for _, s in epochs]
    n_epoch_start = len(starts)
    n_epoch_end = len(ends)

    return {
        "n_params": n_params,
        "best_epoch_valid_auc": max(epoch_auc) if epoch_auc else None,
        "n_evaluations": len(epoch_auc),
        "final_valid_auc": valid[0], "final_valid_logloss": valid[1],
        "final_test_auc": test[0], "final_test_logloss": test[1],
        "n_epochs": n_epoch_end + (1 if re.search(r"Epoch==?\d+ early stop", txt) else 0),
        "n_epoch_start_lines": n_epoch_start,
        "n_epoch_end_lines": n_epoch_end,
        "epoch_seconds": secs,
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
        "min_epoch_seconds": min(secs) if secs else None,
        "max_epoch_seconds": max(secs) if secs else None,
    }


def is_run_complete(r, expect_params):
    """[PATCH P3] 一个 run 算「已完成」的判据。

    原版只看 train.log 是否**非空**就 skip。于是：跑崩留下的 traceback 也是非空，
    下次运行会被当成「已完成」永久跳过，并在 summary 里写成 param_check=OK、
    所有指标 None —— 这是最危险的一类静默失败（已实测复现）。
    这里改成：必须同时拿到 valid AUC、test AUC 与**正确**的参数量，才算完成。
    """
    return (r.get("final_test_auc") is not None
            and r.get("final_valid_auc") is not None
            and r.get("n_params") == expect_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FCN 仓库路径")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--data-root", required=True, help="含 Criteo_x4_10_h5 的目录")
    ap.add_argument("--seeds", required=True, help="逗号分隔，如 2025,2026,...")
    ap.add_argument("--exp-layers", type=int, default=4)
    ap.add_argument("--lin-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--early-stop", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    expect_params = param_formula(args.exp_layers, args.lin_layers)
    print("期望参数量（闭式公式）= %d" % expect_params, flush=True)
    print("种子列表（E3 必须复用同一份）= %s" % seeds, flush=True)

    runs_dir = os.path.join(args.workdir, "runs", "e0_baseline")
    os.makedirs(runs_dir, exist_ok=True)
    # 把种子列表落盘，供 E3 强制复用
    with open(os.path.join(args.workdir, "runs", "seed_list.json"), "w") as f:
        json.dump({"seeds": seeds, "exp_layers": args.exp_layers,
                   "lin_layers": args.lin_layers, "expect_params": expect_params}, f, indent=2)

    summary = []
    for seed in seeds:
        run_id = "e0_exp%d_lin%d_seed%d" % (args.exp_layers, args.lin_layers, seed)
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")
        run_seconds = None

        # [PATCH P3] 只有「真正拿到正确结果」才 skip；否则归档旧日志后重跑。
        need_run = True
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            if is_run_complete(parse_log(log_path), expect_params):
                print("[skip] %s 已完成（已有正确结果）" % run_id, flush=True)
                need_run = False
            else:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                bad = log_path + ".failed-" + stamp
                os.replace(log_path, bad)
                print("[rerun] %s 上次未产出有效结果（旧日志 -> %s）"
                      % (run_id, os.path.basename(bad)), flush=True)
                shutil.rmtree(os.path.join(run_dir, "checkpoints"), ignore_errors=True)

        if need_run:
            # 用作者的 model_config 模板，覆盖 seed 与层数
            cfg_dir = os.path.join(run_dir, "config")
            os.makedirs(cfg_dir, exist_ok=True)
            for fn in ("dataset_config.yaml", "model_config.yaml"):
                src = os.path.join(args.repo, "config", fn)
                shutil.copy(src, os.path.join(cfg_dir, fn))

            import yaml
            mp = os.path.join(cfg_dir, "model_config.yaml")
            mc = yaml.safe_load(open(mp, encoding="utf-8"))
            base = dict(mc["FCNv2_Criteo"])          # 拷贝，避免与模板共享引用
            base["seed"] = seed
            base["exp_num_layers"] = args.exp_layers
            base["lin_num_layers"] = args.lin_layers
            base["epochs"] = args.epochs
            base["early_stop_patience"] = args.early_stop
            base["batch_size"] = args.batch_size
            base["model_root"] = os.path.join(run_dir, "checkpoints") + "/"
            mc["E0_%s" % run_id] = base
            yaml.safe_dump(mc, open(mp, "w", encoding="utf-8"), allow_unicode=True)

            dp = os.path.join(cfg_dir, "dataset_config.yaml")
            dc = yaml.safe_load(open(dp, encoding="utf-8"))
            # [PATCH P1] 仓库 dataset_config.yaml **没有** Criteo_x4_10_h5 这个键
            # （只有 Criteo_x4_001 / Avazu_x4 / KDD12_x10 ...）。
            # 原版直接 dc["Criteo_x4_10_h5"]["data_root"] = ... => KeyError 立即崩溃。
            # 这里按作者日志 Params 的口径补建：data_format=h5、min_categr_count=10、
            # feature_cols/label_col 沿用仓库里唯一的 Criteo 定义（39 字段 +
            # convert_to_bucket，与作者日志 feature specs 逐字段一致）。
            if DATASET_KEY not in dc:
                ref = dc[DATASET_KEY_FALLBACK]
                dc[DATASET_KEY] = {
                    "data_format": "h5",
                    "feature_cols": ref["feature_cols"],
                    "label_col": ref["label_col"],
                    "min_categr_count": 10,
                    "train_data": os.path.join(args.data_root, DATASET_KEY, "train.h5"),
                    "valid_data": os.path.join(args.data_root, DATASET_KEY, "valid.h5"),
                    "test_data": os.path.join(args.data_root, DATASET_KEY, "test.h5"),
                }
                print("  [patch] dataset_config 缺少 %s，已按 %s 的字段定义补建"
                      % (DATASET_KEY, DATASET_KEY_FALLBACK), flush=True)
            dc[DATASET_KEY]["data_root"] = args.data_root.rstrip("/") + "/"
            for k in ("train_data", "valid_data", "test_data"):
                fn2 = os.path.basename(dc[DATASET_KEY][k]) if k in dc[DATASET_KEY] else k.replace("_data", ".h5")
                dc[DATASET_KEY][k] = os.path.join(
                    args.data_root.rstrip("/"), DATASET_KEY, fn2)
            yaml.safe_dump(dc, open(dp, "w", encoding="utf-8"), allow_unicode=True)

            # [PATCH P2] 写兼容启动器并记录 md5（可追溯、可复现）
            launcher = os.path.join(run_dir, "_run_expid_compat.py")
            with open(launcher, "w", encoding="utf-8") as lf:
                lf.write(COMPAT_LAUNCHER)
            launch_md5 = hashlib.md5(open(launcher, "rb").read()).hexdigest()

            cmd = [sys.executable, "-X", "utf8", launcher,
                   os.path.join(args.repo, "run_expid.py"),
                   "--config", cfg_dir + "/", "--expid", "E0_%s" % run_id, "--gpu", "0"]
            print("[run] %s  (compat launcher md5=%s)" % (run_id, launch_md5), flush=True)
            t0 = time.time()
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, cwd=args.repo, stdout=lf,
                                      stderr=subprocess.STDOUT)
            run_seconds = round(time.time() - t0, 1)
            print("      完成，用时 %.1f 分钟，退出码 %d" % (run_seconds / 60, proc.returncode),
                  flush=True)
            if proc.returncode != 0:
                # [PATCH P3] 退出码非 0 必须显式报出来，不能静默继续
                print("  !! 本 run 退出码 %d，日志尾部：" % proc.returncode, flush=True)
                tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-8:]
                for line in tail:
                    print("     | " + line, flush=True)

        r = parse_log(log_path)
        r["seed"] = seed
        r["run_id"] = run_id
        r["run_seconds"] = run_seconds
        # 参数量硬校验
        if r["n_params"] is None:
            r["param_check"] = "NO_PARAMS"      # [PATCH P3] 没跑起来 ≠ OK
        elif r["n_params"] != expect_params:
            r["param_check"] = "MISMATCH"
        else:
            r["param_check"] = "OK"
        summary.append(r)
        print("      valid=%s test=%s params=%s (%s) mean_epoch=%.1fs" %
              (r["final_valid_auc"], r["final_test_auc"], r["n_params"], r["param_check"],
               r["mean_epoch_seconds"] or 0.0), flush=True)

        with open(os.path.join(args.workdir, "runs", "e0_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # [PATCH P3] 参数量 MISMATCH / 没跑起来 => 立刻中止，不要白跑剩下的种子（每个 ~76 分钟）
        if r["param_check"] == "MISMATCH":
            print("\n!!! 参数量不符（实测 %s，期望 %d），按纪律立即中止，"
                  "剩余种子不再运行 !!!" % (r["n_params"], expect_params), flush=True)
            sys.exit(2)
        if r["param_check"] != "OK":
            print("\n!!! 本 run 没有产出有效结果（见上方日志尾部），中止 !!!", flush=True)
            sys.exit(3)

    # ---- 统计 ----
    import statistics
    va = [r["final_valid_auc"] for r in summary if r["final_valid_auc"]]
    ta = [r["final_test_auc"] for r in summary if r["final_test_auc"]]
    es = [r["mean_epoch_seconds"] for r in summary if r["mean_epoch_seconds"]]
    out = {"n_seeds": len(summary),
           "expect_params": expect_params,
           "valid_mean": statistics.mean(va) if va else None,
           "valid_std": statistics.stdev(va) if len(va) > 1 else None,
           "valid_min": min(va) if va else None,
           "valid_max": max(va) if va else None,
           "test_mean": statistics.mean(ta) if ta else None,
           "test_std": statistics.stdev(ta) if len(ta) > 1 else None,
           "epoch_seconds_mean": statistics.mean(es) if es else None,
           "runs": summary}
    with open(os.path.join(args.workdir, "runs", "e0_stats.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=========== E0 汇总 ===========")
    if va:
        print("valid AUC : mean %.6f  std %.6f  [%.6f, %.6f]" %
              (out["valid_mean"], out["valid_std"] or 0, out["valid_min"], out["valid_max"]))
    if ta:
        print("test  AUC : mean %.6f  std %.6f" % (out["test_mean"], out["test_std"] or 0))
    if es:
        print("每 epoch 实测: %.1f s（DCU，含 valid 评估）" % out["epoch_seconds_mean"])
    print("作者基准  : valid 0.815627 / test 0.816145 (_016_cde6d2ff.log)")
    if out["valid_std"]:
        print("实测 σ(valid) = %.6f  → 这决定了 E3 能分辨多大的提升" % out["valid_std"])
        for d in (0.0005, 0.001, 0.002):
            n_needed = int((2.8 * out["valid_std"] / d) ** 2 + 0.999)
            print("   要分辨 ΔAUC=%.4f 需配对 %d 个种子" % (d, max(n_needed, 1)))
    print("================================")


if __name__ == "__main__":
    main()


# ============================== PATCH LOG ==============================
# 原文件：tools/run_e0_baseline.py（199 行，md5 ec60bae421d1aa71dfaa7dcb768a66c4）
# 本文件：5 处补丁，全部为「让它能跑 / 不静默骗人」，不改任何实验设计。
#
# P1  dataset_config 缺 Criteo_x4_10_h5 键
#     原：dc["Criteo_x4_10_h5"]["data_root"] = ...      -> KeyError 立即崩
#     实测：KeyError: 'Criteo_x4_10_h5'（真实仓库 dataset_config.yaml 只有
#           Avazu_x4/Tenrec_x1/Criteo_x4_001/ML_1M_7_x1/iPinYou_x1_csv/KKBox_csv/KDD12_x10）
#     改：键不存在时按 Criteo_x4_001 的 feature_cols/label_col + min_categr_count=10 补建
#
# P2  fuxictr 2.0.1 没有 delete_model_files
#     原：subprocess 直接跑 run_expid.py -> ImportError，每个 run 秒退
#     实测：ImportError: cannot import name 'delete_model_files' from 'fuxictr.utils'
#           （已核对 PyPI wheel 内 fuxictr/utils.py 源码，确实无此函数）
#     改：写一个 run 级启动器，执行前把该符号补进 fuxictr.utils，再 runpy 执行
#         run_expid.py。不参与计算、不改仓库文件、每次记录 md5。
#
# P3  失败 run 被静默当成「已完成」
#     原：train.log 非空就 skip；退出码不检查；参数量为 None 时 param_check 仍写 "OK"
#     实测：放一个内容是 traceback 的 train.log，驱动打印 "[skip] 已完成"，
#           summary 里 8 个字段全 null 而 param_check="OK"，MISMATCH 告警永不触发
#     改：只有「valid AUC + test AUC + 参数量正确」才算完成；否则归档旧日志并重跑；
#         param_check 增加 NO_PARAMS；MISMATCH 立即 sys.exit(2) 中止后续种子
#
# P4  单轮耗时 / 轮数静默算错（captain 也独立发现并确认）
#     原：用 "Epoch=N start" 行做时间戳差分。但 fuxictr fit() 只在开头打一次
#         "Epoch=1 start"，之后每轮只有 "Epoch=N end"。
#     实测：作者 19 轮日志 "Epoch= start" 只有 1 行、"Epoch= end" 有 18 行
#           => 原版 n_epochs 恒为 1、mean_epoch_seconds 恒为 None。
#     改：epoch_1 = end_1 - start_1；epoch_N = end_N - end_{N-1}；
#         同时输出 n_epoch_start_lines / n_epoch_end_lines / epoch_seconds 全序列，
#         让「end 行比实际轮数少 1（早停轮不打 end）」这件事在数据里可见，而不是被抹平。
#
# P4b 上面这个修法第一版**也失效了**，原因值得记一笔：正则用 `^时间戳` 锚行首，
#     但我们自己跑时把 stderr 并进了 train.log（否则 traceback 会丢），而 tqdm 进度条
#     写 stderr 且**不换行**，并流后物理行变成
#         100%|███| 1/1 [00:18<00:00, 18.58s/it]2026-11-18 21:38:58,552 ... Epoch=1 end
#     时间戳不在行首 -> 一条都匹配不到（实测 n_epoch_end_lines=0，epoch_seconds=[]，
#     依旧静默返回空）。改为不锚行首、允许中间夹杂任意非换行字符。
#     根因与 captain 报的坑同族：**解析失败时返回空列表而不是抛异常**。
#     因此本版把 n_epoch_start_lines / n_epoch_end_lines 一并写进产出，
#     用「行数为 0」自证解析是否生效，不让下游靠肉眼发现 mean 是 None。
#
# P5  base = mc["FCNv2_Criteo"] 是共享引用，写回 mc 后会被 yaml 打成 &id001 锚点，
#     下次读回同一个 dict 对象。功能上无害，但为了让每次 run 的配置可独立阅读，
#     改成 base = dict(mc["FCNv2_Criteo"]) 拷贝一份。
# =======================================================================
