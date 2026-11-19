#!/usr/bin/env python3
"""E0 基线复现 + 种子噪声实测（在 DCU 上跑真实 Criteo 数据）。

设计要点（为什么这么写）：
1. **配对设计的基础**：所有种子写成列表落盘，E3 改进实验必须用同一份列表，
   这样种子效应可相消，把需要的样本量从「独立 16」降到「配对 8」。
2. **每个 run 独立目录**：不覆盖，便于任何一次结果都能追溯。
3. **参数量自动校验**：每个 run 都用闭式公式核对实际参数量，对不上立刻报错，
   防止「配置没生效却不知道」。
4. **valid / test 严格分列**：早停是在 valid 上选的模型，所以 test 系统性偏高，
   两者混列会得出反向结论（这一点已被实际日志验证过）。

用法：
  python3 run_e0_baseline.py --repo <FCN> --workdir <dir> --seeds 2025..2032
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


def param_formula(exp_layers, lin_layers, d=624, emb_dim=16, vocab_total=910747):
    """FCNv2 闭式参数量。与 tools/param_formula.py 同源，用于独立校验。"""
    per_layer = d * (d // 2) + (d // 2) + (d // 2) + d
    one_net = (exp_layers + lin_layers) * per_layer + (d + 1)
    return vocab_total * emb_dim + 2 * one_net


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

    # [P4] fuxictr fit() **只在第一个 epoch 打 "Epoch=1 start"**，之后每轮只打
    # "Epoch=N end"。用 start 行差分 => secs 恒为空、n_epochs 恒为 1、
    # mean_epoch_seconds 恒为 None，而且**不报错**（静默失效）。
    # 正确算法：epoch_1 = end_1 - start_1；epoch_N = end_N - end_{N-1}。
    #
    # [P4b] 正则**不能**锚行首 `^时间戳`：tqdm 进度条写 stderr 且不换行，并流进
    # train.log 后物理行是
    #   100%|███| 1/1 [00:18<00:00, 18.58s/it]2025-05-13 06:38:23,835 ... Epoch=1 end
    # 时间戳不在行首 => 一条都匹配不到（实测 n_epoch_end_lines=0，依旧静默返回空）。
    _TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)"
    from datetime import datetime
    starts = re.findall(_TS + r" .*?Epoch=(\d+) start ", txt)
    ends = re.findall(_TS + r" .*?Epoch=(\d+) end ", txt)
    secs = []
    prev = datetime.strptime(starts[0][0], "%Y-%m-%d %H:%M:%S,%f") if starts else None
    for _ts, _no in ends:
        _cur = datetime.strptime(_ts, "%Y-%m-%d %H:%M:%S,%f")
        if prev is not None:
            secs.append((_cur - prev).total_seconds())
        prev = _cur
    _early = re.search(r"Epoch==?\d+ early stop", txt)
    # 早停那一轮只打 "early stop"、不打 "end"，所以实际轮数 = end 行数 + 是否早停
    n_epoch = len(ends) + (1 if _early else 0)
    early_m = re.search(r"Epoch==?(\d+) early stop", txt)

    # [P6] lr 衰减时序。日志的真实顺序是：
    #   Evaluation @epoch 14 - batch 4477:
    #   [Metrics] AUC: 0.812342 - logloss: 0.439460
    #   Monitor(max)=0.812342 STOP!
    #   Reduce learning rate on plateau: 0.000100
    #   ************ Epoch=14 end ************
    # 注意「Epoch=14 end」在这条消息**之后**才打印（因为 checkpoint_and_earlystop
    # 在 train_epoch 内被调用，而 end 行在 fit 的循环体末尾）。
    # 所以归属必须用**紧邻其前的 `Evaluation @epoch N`**，不能用 end 行。
    # （第一版用 end 行，得到 [13,17,18]，全错一位 —— 已修）
    lr_decay_epochs, lr_decay_values = [], []
    for _m in re.finditer(r"Reduce learning rate on plateau: ([\d.eE+-]+)", txt):
        lr_decay_values.append(float(_m.group(1)))
        _prev = None
        for _em in re.finditer(r"Evaluation @epoch (\d+) - batch", txt[:_m.start()]):
            _prev = int(_em.group(1))
        lr_decay_epochs.append(_prev)
    return {
        "n_params": n_params,
        "best_epoch_valid_auc": max(epoch_auc) if epoch_auc else None,
        "n_evaluations": len(epoch_auc),
        "final_valid_auc": valid[0], "final_valid_logloss": valid[1],
        "final_test_auc": test[0], "final_test_logloss": test[1],
        "n_epochs": n_epoch,
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
        "min_epoch_seconds": min(secs) if secs else None,
        "max_epoch_seconds": max(secs) if secs else None,
        # 自证字段：解析没生效时这些会是 0 / []，一眼可见，
        # 不必靠肉眼发现 mean_epoch_seconds 是 None
        "n_epoch_start_lines": len(starts),
        "n_epoch_end_lines": len(ends),
        "epoch_seconds": secs,
        # [P6] lr 衰减时序 —— 「三段式轨迹」的量化指标
        # fuxictr base_model.checkpoint_and_earlystop：
        #   if monitor_value < best + min_delta(=1e-6):   # 本轮没提升
        #       "Monitor(max)=... STOP!" -> if reduce_lr_on_plateau: lr_decay()
        # 实测作者 _016：ep13=0.812382 -> ep14=0.812342（低）-> 衰减 1e-3->1e-4
        #                -> ep15 从 0.812342 跳到 0.815144（+0.0028，占总提升 26%）
        # 因为 min_delta 只有 1e-6，而我们与作者的逐轮 Δ 在 1e-5~1e-4，
        # 所以「哪一轮触发衰减」可能差 1-2 轮，最终 AUC 会因此差 1e-3 量级。
        # 记录这个时序，才能区分「链路不对」与「分岔点漂移」。
        "lr_decay_epochs": lr_decay_epochs,
        "lr_decay_values": lr_decay_values,
        "early_stop_epoch": (int(early_m.group(1)) if early_m else None),
    }


def is_run_complete(r, expect_params):
    """[P3] 一个 run 算「已完成」的判据：三项齐备，缺一即重跑。"""
    return (r.get("final_valid_auc") is not None
            and r.get("final_test_auc") is not None
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

        # [P3] 原版只看 train.log 是否**非空**就 skip。跑崩留下的 traceback 也是
        # 非空，于是失败的种子被永久当成「已完成」，never retried，而 summary 里
        # 只写 failed=false / NO_PARAMS，统计时静默少算一个 n。实测已复现。
        # 改成：必须同时有 valid AUC、test AUC 且参数量正确，才算完成。
        need_run = True
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            if is_run_complete(parse_log(log_path), expect_params):
                print("[skip] %s 已完成（valid/test/参数量三项齐备）" % run_id, flush=True)
                need_run = False
            else:
                _stamp = time.strftime("%Y%m%d-%H%M%S")
                _bad = log_path + ".failed-" + _stamp
                os.replace(log_path, _bad)
                shutil.rmtree(os.path.join(run_dir, "checkpoints"), ignore_errors=True)
                print("[rerun] %s 上次未产出有效结果，旧日志归档为 %s"
                      % (run_id, os.path.basename(_bad)), flush=True)
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
            base = mc["FCNv2_Criteo"]
            base["seed"] = seed
            base["exp_num_layers"] = args.exp_layers
            base["lin_num_layers"] = args.lin_layers
            base["epochs"] = args.epochs
            base["early_stop_patience"] = args.early_stop
            base["batch_size"] = args.batch_size
            base["model_root"] = os.path.join(run_dir, "checkpoints") + "/"
            mc["E0_%s" % run_id] = base
            yaml.safe_dump(mc, open(mp, "w", encoding="utf-8"), allow_unicode=False)

            dp = os.path.join(cfg_dir, "dataset_config.yaml")
            dc = yaml.safe_load(open(dp, encoding="utf-8"))

            # 【重要坑，已实测确认】作者仓库的 dataset_config.yaml 里**没有**
            # `Criteo_x4_10_h5` 这一项（只有 Criteo_x4_001，且是 csv 格式）。
            # 但 model_config.yaml 里 FCNv2_Criteo 的 dataset_id 正是 Criteo_x4_10_h5。
            # 作者自己的日志显示他跑的是 data_format=h5，说明他本地的 config 有这一项，
            # 但没提交到仓库。所以必须在这里补出来，否则 load_config 会报
            # "dataset_id=Criteo_x4_10_h5 is not found in config"。
            if "Criteo_x4_10_h5" not in dc:
                src_ds = dc["Criteo_x4_001"]
                h5dir = os.path.join(args.data_root.rstrip("/"), "Criteo_x4_10_h5")
                dc["Criteo_x4_10_h5"] = {
                    "data_format": "h5",
                    "data_root": args.data_root.rstrip("/") + "/",
                    "feature_cols": src_ds["feature_cols"],
                    "label_col": src_ds["label_col"],
                    "min_categr_count": src_ds["min_categr_count"],
                    "train_data": os.path.join(h5dir, "train.h5"),
                    "valid_data": os.path.join(h5dir, "valid.h5"),
                    "test_data": os.path.join(h5dir, "test.h5"),
                }
                print("  [fix] dataset_config 缺 Criteo_x4_10_h5，已按 h5 口径补出", flush=True)
            else:
                dc["Criteo_x4_10_h5"]["data_root"] = args.data_root.rstrip("/") + "/"
                for k in ("train_data", "valid_data", "test_data"):
                    fn2 = os.path.basename(dc["Criteo_x4_10_h5"][k])
                    dc["Criteo_x4_10_h5"][k] = os.path.join(
                        args.data_root.rstrip("/"), "Criteo_x4_10_h5", fn2)
            # 注意：必须 allow_unicode=False 或保证全 ASCII，
            # 因为 fuxictr 2.0.1 的 load_config 用 open(config) 不指定 encoding，
            # 非 ASCII 在 GBK/C locale 下会 UnicodeDecodeError（modeler 实测踩到）。
            yaml.safe_dump(dc, open(dp, "w", encoding="utf-8"), allow_unicode=False)

            cmd = [sys.executable, "-X", "utf8",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_expid_fixed.py"),
                   "--config", cfg_dir + "/", "--expid", "E0_%s" % run_id,
                   "--gpu", "0", "--repo", args.repo,
                   "--result-json", os.path.join(run_dir, "result.json")]
            print("[run] %s" % run_id, flush=True)
            t0 = time.time()
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, cwd=args.repo, stdout=lf,
                                      stderr=subprocess.STDOUT)
            dt = time.time() - t0
            print("      完成，用时 %.1f 分钟，退出码 %d" % (dt / 60, proc.returncode),
                  flush=True)
            if proc.returncode != 0:
                # 【重要】绝不能把失败的 run 当成功继续。之前这里没检查返回码，
                # 导致失败的 run 被记成 valid=0.0 还打 OK，是严重缺陷。
                print("  !! 该 run 失败（退出码 %d），已中止后续种子。" % proc.returncode,
                      flush=True)
                print("     请查看 %s" % log_path, flush=True)
                summary.append({"seed": seed, "run_id": run_id,
                                "failed": True, "returncode": proc.returncode})
                with open(os.path.join(args.workdir, "runs", "e0_summary.json"), "w") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                sys.exit(2)

        r = parse_log(log_path)
        r["seed"] = seed
        r["run_id"] = run_id
        r["failed"] = False
        # 参数量硬校验：如果解析不到，说明 run 有问题，不要静默通过
        if r["n_params"] is None:
            print("  !! 日志里解析不到参数量，该 run 视为无效", flush=True)
            r["param_check"] = "NO_PARAMS"
        elif r["n_params"] != expect_params:
            print("  !! 参数量不符：实测 %d，期望 %d" % (r["n_params"], expect_params),
                  flush=True)
            r["param_check"] = "MISMATCH"
        else:
            r["param_check"] = "OK"
        summary.append(r)
        print("      valid=%.6f test=%.6f params=%s (%s)" %
              (r["final_valid_auc"] or 0, r["final_test_auc"] or 0,
               r["n_params"], r["param_check"]), flush=True)

        with open(os.path.join(args.workdir, "runs", "e0_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # [P3b] 参数量不符 / 没跑起来 => 立即中止。原版只打印一行就继续，真实数据
        # 76 min/seed，8 个种子会白烧 10 小时才在最后发现配置没生效。
        if r["param_check"] == "MISMATCH":
            print("\n!!! 参数量不符（实测 %s，期望 %d），按纪律立即中止，"
                  "剩余种子不再运行 !!!" % (r["n_params"], expect_params), flush=True)
            sys.exit(2)
        if r["param_check"] != "OK":
            print("\n!!! 本 run 未产出有效结果，已中止（日志：%s）!!!" % log_path, flush=True)
            sys.exit(3)

    # ---- 统计 ----
    import statistics
    va = [r["final_valid_auc"] for r in summary if r["final_valid_auc"]]
    ta = [r["final_test_auc"] for r in summary if r["final_test_auc"]]
    # [P3] n_seeds 必须区分「声明跑了几个」与「真正有几个有效」。
    # 原版把 n_seeds 写成 len(summary)，一旦某 run 无效，报告里 n 会虚报。
    _valid_runs = [r for r in summary if r.get("final_valid_auc") is not None
                   and r.get("final_test_auc") is not None]
    out = {"n_seeds": len(_valid_runs),
           "n_seeds_attempted": len(summary),
           "n_seeds_invalid": len(summary) - len(_valid_runs),
           "invalid_runs": [r["run_id"] for r in summary
                            if r.get("final_valid_auc") is None],
           "expect_params": expect_params,
           "valid_mean": statistics.mean(va) if va else None,
           "valid_std": statistics.stdev(va) if len(va) > 1 else None,
           "valid_min": min(va) if va else None,
           "valid_max": max(va) if va else None,
           "test_mean": statistics.mean(ta) if ta else None,
           "test_std": statistics.stdev(ta) if len(ta) > 1 else None,
           "runs": summary}
    with open(os.path.join(args.workdir, "runs", "e0_stats.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=========== E0 汇总 ===========")
    if va:
        print("valid AUC : mean %.6f  std %.6f  [%.6f, %.6f]" %
              (out["valid_mean"], out["valid_std"] or 0, out["valid_min"], out["valid_max"]))
    if ta:
        print("test  AUC : mean %.6f  std %.6f" % (out["test_mean"], out["test_std"] or 0))
    print("作者基准  : valid 0.815627 / test 0.816145 (_016_cde6d2ff.log)")
    if out["valid_std"]:
        print("实测 σ(valid) = %.6f  → 这决定了 E3 能分辨多大的提升" % out["valid_std"])
        for d in (0.0005, 0.001, 0.002):
            n_needed = int((2.8 * out["valid_std"] / d) ** 2 + 0.999)
            print("   要分辨 ΔAUC=%.4f 需配对 %d 个种子" % (d, max(n_needed, 1)))
    print("================================")


if __name__ == "__main__":
    main()

