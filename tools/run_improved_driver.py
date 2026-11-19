#!/usr/bin/env python3
"""改进实验（E3/E4-A/E4-B）的 driver：种子从 seed_list.json 读，不是手写循环。

=============================================================================
为什么需要这个 wrapper
-----------------------------------------------------------------------------
`run_improved.py` 只有 `--seed`（单值），没有 `--seeds`。所以跑多种子只能手写：
    for S in 2025 2026 2027 2028 ...; do
      python3 run_improved.py ... --seed $S --out runs/e3/e3_seed$S
    done

手写循环有三个已知风险，都已在别处真实发生过：
  1. **漏一个种子** -> 配对设计退化成独立样本，需要的样本量翻倍，而报告里看不出来。
     项目纪律写的是「E3 必须复用 E0 的同一份种子列表」，但此前**没有任何工具强制**它。
  2. **目录名不带 seed<N>** -> `paired_analysis.py` 找不到那个 run，静默少算。
  3. **配置没生效照跑** -> 8 种子里有 6 个配置的参数量完全相同（16,532,562），
     参数量闸门对它们**是盲的**。必须靠 `verify_improved_configs.py` 的
     层数/tri_mode/lca 三维才能判断。

本 wrapper 把这三件事都变成代码强制：
  * 种子**只从** `runs/seed_list.json` 读（不读就把 `--seeds` 当显式覆盖，并记入审计）
  * 每个 run 的目录名**强制**含 `_seed<N>`
  * 每个 run 结束后**立即**做参数量硬校验，不符就停
  * 每个 run 结束后**立即**做配对性检查（缺种子当场报）

用法：
    # E3：Tri-BCE 软权重，配对 E0 的 8 个种子
    python3 run_improved_driver.py \
        --configs $P/code/configs --expid FCNv2_C_soft_smoothmax \
        --workdir $P/runs --data-root $P/data \
        --out-subdir e3_soft --epochs 20

    # E4-B：LCA 扫描，每档 1 个种子（参数量是确定性指标，n=1 够）
    python3 run_improved_driver.py \
        --configs $P/code/configs --expid FCNv2_C_9L_r050 \
        --workdir $P/runs --data-root $P/data \
        --out-subdir e4b_9L_r050 --seeds 2025 --epochs 20

    # 只演练不真跑（打印将执行的命令）
    python3 run_improved_driver.py ... --dry-run

退出码：0=全部成功；2=参数量不符（立即停）；3=某 run 失败（立即停）；4=配对性破坏
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# 每个 expid 的期望 (n_params, exp, lin, tri_mode, lca)。与 verify_improved_configs.py 同源。
EXPECTED = {
    "FCNv2_C_base":                  (16532562, 4, 1, "hard",      0.5),
    "FCNv2_C_soft_smoothmax":        (16532562, 4, 1, "smoothmax", 0.5),
    "FCNv2_C_soft_softplus":         (16532562, 4, 1, "softplus",  0.5),
    "FCNv2_C_soft_ratio":            (16532562, 4, 1, "ratio",     0.5),
    "FCNv2_C_soft_smoothmax_anneal": (16532562, 4, 1, "smoothmax", 0.5),
    "FCNv2_C_abl_no_tribce":         (16532562, 4, 1, "plain",     0.5),
    "FCNv2_C_abl_exp0":              (16532562, 0, 5, "hard",      0.5),
    "FCNv2_C_abl_lin0":              (16532562, 5, 0, "hard",      0.5),
    "FCNv2_C_r025":                  (15556002, 4, 1, "hard",      0.25),
    "FCNv2_C_r075":                  (17509122, 4, 1, "hard",      0.75),
    "FCNv2_C_r100":                  (18473202, 4, 1, "hard",      1.0),
    "FCNv2_C_9L_r025":               (16342242, 4, 5, "hard",      0.25),
    "FCNv2_C_9L_r050":               (18100050, 4, 5, "hard",      0.5),
    "FCNv2_C_9L_r075":               (19857858, 4, 5, "hard",      0.75),
    "FCNv2_C_9L_r100":               (21593202, 4, 5, "hard",      1.0),
}


def _read_cfg_layers(configs, expid):
    """从 model_config.yaml 里读某个 expid 的 (exp_num_layers, lin_num_layers)。

    为什么不从 result.json 读：实测 `run_improved.py` 的 result.json **不含**这两个键。
    只看 result.json 会让「配置没生效」静默通过（.get() 返回 None）。
    配置源头是唯一可靠的地方——但注意：这是「配置写了什么」，
    而「模型实际用了什么」要靠参数量与 (tri, lca) 联合佐证。
    """
    import yaml
    for cand in (os.path.join(configs, "model_config.yaml"),
                 os.path.join(configs, "model_config")):
        paths = [cand] if os.path.isfile(cand) else sorted(glob.glob(os.path.join(cand, "*.yaml")))
        for p in paths:
            try:
                d = yaml.safe_load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict) and expid in d:
                blk = d[expid]
                return blk.get("exp_num_layers"), blk.get("lin_num_layers")
    return None, None


def load_seeds(workdir, explicit):
    p = os.path.join(workdir, "runs", "seed_list.json")
    if not os.path.exists(p):
        p2 = os.path.join(workdir, "seed_list.json")
        p = p2 if os.path.exists(p2) else p
    if not os.path.exists(p):
        if explicit:
            print("  [warn] 找不到 seed_list.json，用 --seeds 的显式列表 %s" % explicit)
            print("         -> 这会**破坏与 E0 的配对性**，报告里必须说明原因")
            return [int(s) for s in explicit.split(",") if s.strip()], None
        sys.exit("!! 找不到 seed_list.json：%s\n"
                 "   E3/E4-A 必须复用 E0 的种子列表（配对设计）。\n"
                 "   若确实要用别的种子，请显式给 --seeds 并记录原因。" % p)
    d = json.load(open(p, encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    if explicit:
        want = [int(s) for s in explicit.split(",") if s.strip()]
        if want != seeds:
            print("  [warn] --seeds=%s 与 seed_list.json=%s 不同" % (want, seeds))
            print("         -> 用 --seeds 的；但配对性会以它为准，请确认这是有意的")
            return want, p
    return seeds, p


def run_one(args, expid, seed, out_dir, log_path, dry):
    """跑单个种子。返回 (ok, n_params, detail)。"""
    cmd = [sys.executable, "-X", "utf8",
           os.path.join(args.code_src, "src", "run_improved.py"),
           "--config", args.configs, "--expid", expid, "--gpu", str(args.gpu),
           "--seed", str(seed), "--epochs", str(args.epochs),
           "--data_root", args.data_root, "--out", out_dir]
    if args.model_root:
        cmd += ["--model_root", args.model_root]
    if args.extra:
        cmd += args.extra.split()

    if dry:
        print("    DRY: %s" % " ".join(cmd))
        return True, None, "dry-run"

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    rj = os.path.join(out_dir, "result.json")
    n_params = None
    if os.path.exists(rj):
        try:
            n_params = json.load(open(rj, encoding="utf-8")).get("n_params_total")
        except Exception:
            pass
    ok = (proc.returncode == 0 and n_params is not None)
    return ok, n_params, "%.1fs rc=%d params=%s" % (dt, proc.returncode, n_params)


def main(argv=None):
    ap = argparse.ArgumentParser(description="改进实验 driver（种子来自 seed_list.json）")
    ap.add_argument("--configs", required=True, help="<project>/code/configs")
    ap.add_argument("--expid", required=True)
    ap.add_argument("--workdir", required=True, help="项目根（含 runs/）")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-subdir", required=True, help="runs/ 下的子目录名，如 e3_soft")
    ap.add_argument("--seeds", default=None, help="显式种子（会破坏配对，勿轻用）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-root", default=None,
                    help="建议显式指定，避免 checkpoint 落到 cwd（实测会落进项目根）")
    ap.add_argument("--extra", default=None, help="附加参数，如 '--tri_bce_alpha 2.0'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # expid 必须在基准表里，否则没有参数量可校验
    if args.expid not in EXPECTED:
        sys.exit("!! expid %r 不在基准表里，无法校验参数量。可用的：\n  %s"
                 % (args.expid, "\n  ".join(sorted(EXPECTED))))
    exp_params, exp_exp, exp_lin, exp_tri, exp_lca = EXPECTED[args.expid]

    # --configs 通常是 <project>/code/configs，而 src/ 在 <project>/code/src。
    # 所以 code_src 就是 configs 的**父目录**（一层），别多取。
    args.code_src = os.path.dirname(os.path.abspath(args.configs.rstrip("/")))
    if not os.path.isdir(os.path.join(args.code_src, "src")):
        sys.exit("!! %s 下没有 src/，--configs 应指向 <project>/code/configs（当前解析到 %s）"
                 % (args.code_src, args.configs))

    seeds, seed_file = load_seeds(args.workdir, args.seeds)
    base = os.path.join(args.workdir, "runs", args.out_subdir)
    os.makedirs(base, exist_ok=True)

    print("=" * 88)
    print("改进实验 driver")
    print("  expid     : %s" % args.expid)
    print("  期望      : params=%d, exp/lin=%d/%d, tri=%s, lca=%s"
          % (exp_params, exp_exp, exp_lin, exp_tri, exp_lca))
    print("  种子来源  : %s" % (seed_file or "显式 --seeds"))
    print("  种子列表  : %s" % seeds)
    print("  输出目录  : %s" % base)
    print("=" * 88)

    results = []
    for seed in seeds:
        # 目录名强制含 _seed<N>，这样 paired_analysis.py 一定能找到
        run_dir = os.path.join(base, "%s_seed%d" % (args.out_subdir, seed))
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "driver.log")

        print("\n[seed %d] -> %s" % (seed, run_dir), flush=True)
        ok, n_params, detail = run_one(args, args.expid, seed, run_dir, log_path, args.dry_run)
        print("          %s" % detail, flush=True)

        if args.dry_run:
            continue

        # ---- 每个 run 结束立刻做参数量硬校验 ----
        if not ok:
            print("  !! run 失败（%s），立即停。" % detail)
            print("     日志：%s" % log_path)
            tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-8:]
            for line in tail:
                print("     | " + line)
            return 3
        if n_params != exp_params:
            print("  !! 参数量 %s ≠ 期望 %d，立即停（不要白跑剩下的种子）"
                  % (n_params, exp_params))
            return 2

        # ---- 同时核 exp/lin/tri/lca（参数量相同的 8 个配置只能靠这个区分）----
        # 坑：run_improved.py 的 result.json **没有** exp_num_layers / lin_num_layers
        # 这两个键（实测只有 n_params_total/n_params_cross/improvements 等）。
        # 所以不能只读 result.json 的 .get() —— 缺失时会返回 None，
        # 而 None in (None, exp) 恒为 True，会**静默通过**。
        # exp/lin 必须回到配置源头去核：读 <configs>/model_config.yaml 里该 expid 的块。
        try:
            r = json.load(open(os.path.join(run_dir, "result.json"), encoding="utf-8"))
            imp = r.get("improvements", {})
            bad = []

            # (a) 配置源头的 exp/lin
            cfg_exp, cfg_lin = _read_cfg_layers(args.configs, args.expid)
            if cfg_exp is None:
                bad.append("配置里读不到 exp/lin（model_config.yaml 的 %s 块）" % args.expid)
            else:
                if cfg_exp != exp_exp:
                    bad.append("配置 exp=%s≠%d" % (cfg_exp, exp_exp))
                if cfg_lin != exp_lin:
                    bad.append("配置 lin=%s≠%d" % (cfg_lin, exp_lin))

            # (b) 模型实例上记录的层数（若有；run_improved 未写则跳过并提示）
            if "exp_num_layers" in r and r["exp_num_layers"] is not None:
                if r["exp_num_layers"] != exp_exp:
                    bad.append("实例 exp=%s≠%d" % (r["exp_num_layers"], exp_exp))
            if "lin_num_layers" in r and r["lin_num_layers"] is not None:
                if r["lin_num_layers"] != exp_lin:
                    bad.append("实例 lin=%s≠%d" % (r["lin_num_layers"], exp_lin))

            # (c) tri_bce_mode / lca_compress_ratio（result.json 有）
            tri_got = imp.get("tri_bce_mode")
            if tri_got is None:
                bad.append("result.json 缺 improvements.tri_bce_mode（无法核实）")
            elif tri_got != exp_tri:
                bad.append("tri=%s≠%s" % (tri_got, exp_tri))
            lca_got = imp.get("lca_compress_ratio")
            if lca_got is None:
                bad.append("result.json 缺 improvements.lca_compress_ratio（无法核实）")
            elif abs(float(lca_got) - exp_lca) > 1e-9:
                bad.append("lca=%s≠%s" % (lca_got, exp_lca))

            if bad:
                print("  !! 配置未按预期生效：%s -> 立即停" % "; ".join(bad))
                return 2
            print("         配置核对 OK（params=%d / exp=%d / lin=%d / tri=%s / lca=%s）"
                  % (n_params, exp_exp, exp_lin, exp_tri, exp_lca), flush=True)
        except Exception as exc:
            print("  [warn] 读 result.json 做配置核对失败：%s" % exc)

        results.append({"seed": seed, "dir": run_dir, "n_params": n_params})

    if args.dry_run:
        print("\n(dry-run 结束，未真跑)")
        return 0

    # ---- 全部跑完，做配对性检查 ----
    print("\n" + "=" * 88)
    got = sorted(r["seed"] for r in results)
    if seed_file:
        want = seeds
        missing = [s for s in want if s not in got]
        if missing:
            print("配对性破坏：缺 %s" % missing)
            print("这些种子必须补跑，否则 paired_analysis 会拒绝出统计。")
            return 4
    print("全部 %d 个种子完成，配对性完整 ✅" % len(got))
    print("下一步：paired_analysis.py --baseline-dir <E0目录> --improved-dir %s" % base)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
