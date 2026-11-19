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

    n_epoch = len(re.findall(r"Epoch=\d+ start", txt))
    # 每 epoch 秒数：用 start 时间戳差分
    times = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Epoch=(\d+) start",
                       txt, re.M)
    secs = []
    from datetime import datetime
    ts = [(datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f"), int(e)) for t, e in times]
    ts.sort(key=lambda x: x[1])
    for i in range(1, len(ts)):
        secs.append((ts[i][0] - ts[i - 1][0]).total_seconds())
    return {
        "n_params": n_params,
        "best_epoch_valid_auc": max(epoch_auc) if epoch_auc else None,
        "n_evaluations": len(epoch_auc),
        "final_valid_auc": valid[0], "final_valid_logloss": valid[1],
        "final_test_auc": test[0], "final_test_logloss": test[1],
        "n_epochs": n_epoch,
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
    }


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

        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            print("[skip] %s 已完成" % run_id, flush=True)
        else:
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

    # ---- 统计 ----
    import statistics
    va = [r["final_valid_auc"] for r in summary if r["final_valid_auc"]]
    ta = [r["final_test_auc"] for r in summary if r["final_test_auc"]]
    out = {"n_seeds": len(summary),
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

