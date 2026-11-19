#!/usr/bin/env python3
"""E0 开跑后 5 分钟的启动自检。硬断言，不靠人眼看。

=============================================================================
为什么需要这个脚本
-----------------------------------------------------------------------------
E0 要连跑 8 个种子约 10-11 小时。如果启动阶段就有问题，最坏情况是：
  * 白跑 10 小时才发现参数量不对
  * 或者跑完了，`mean_epoch_seconds` 全是 None，报告的「每 epoch 秒数」凭空消失

已经实测发生过的 4 类启动期失败（见 docs/E0执行前验证_runner_20261118.md）：
  P1  dataset_config 缺 Criteo_x4_10_h5 -> KeyError 秒退
  P2  fuxictr 2.0.1 无 delete_model_files -> ImportError 秒退
  P3  跑崩的 run 被当成「已完成」永久跳过
  P4  s/epoch 恒 None（用 Epoch=start 差分，而 fit 只打一次 start）
  +   路径错位：seed_list.json 生成在找得到的地方之外 -> E3 无法复用

本脚本在**第一个种子跑起来之后**执行，逐项硬断言。任何一项失败 -> exit != 0。

=============================================================================
三项核心检查（captain 指定的）
-----------------------------------------------------------------------------
1. `seed_list.json` 在哪、内容对不对（8 个种子、exp/lin、expect_params）
2. 参数量 == 16,532,562
3. 计时解析生效：`n_epochs > 1` 且 `mean_epoch_seconds is not None`

第 3 项是最容易静默失效的，所以做成**硬断言**。

用法：
    # 开跑 5 分钟后
    python3 tools/e0_startup_check.py --workdir <项目根> --seeds 2025,2026,...,2032

    # 只看退出码
    python3 tools/e0_startup_check.py --workdir $P --quiet; echo $?

退出码：0 = 全部通过；1 = 有 FAIL（不要继续等，立刻排查）
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import re
import sys

EXPECT_PARAMS = 16532562
_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)"

# 已知的废弃/对照实现：它们解析失败属预期，记 warn 不记 FAIL。
# 理由（captain 要求）：若不区分，每次自检都会带一个「预期内的 FAIL」，
# 看输出的人（captain / auditor）会误判成真问题；久了就会学会忽略整个 FAIL 列表。
KNOWN_DEPRECATED = {
    "run_e0_baseline.py",        # 双 bug（start 差分 + ^ 锚行首），已被 run_e0_baseline_v2.py 取代
    "run_e0_baseline_patched.py",  # 早期补丁版，仅留作对照
}


# ---------------------------------------------------------------------------
# seed_list.json 可能的位置（兼容 --workdir 的两种语义）
# ---------------------------------------------------------------------------
def find_seed_list(workdir):
    """按可能性从高到低找 seed_list.json，返回 [(path, how)]。"""
    cands = [
        (os.path.join(workdir, "runs", "seed_list.json"), "workdir/runs/（driver 默认写法）"),
        (os.path.join(workdir, "seed_list.json"), "workdir/（e0_guard 旧写法）"),
    ]
    out = [(p, h) for p, h in cands if os.path.exists(p)]
    if not out:
        # 兜底：整个 workdir 下搜（最多 3 层，避免扫到 h5）
        for p in glob.glob(os.path.join(workdir, "**", "seed_list.json"), recursive=True):
            if p.count(os.sep) - workdir.count(os.sep) <= 3:
                out.append((p, "搜出来的（位置非预期，需留意）"))
    return out


def find_run_dirs(workdir):
    """找 e0_baseline 下各 seed 的 run 目录。"""
    cands = [
        os.path.join(workdir, "runs", "e0_baseline"),
        os.path.join(workdir, "e0_baseline"),
    ]
    cands += glob.glob(os.path.join(workdir, "**", "e0_baseline"), recursive=True)
    for d in cands:
        if os.path.isdir(d):
            runs = [x for x in sorted(glob.glob(os.path.join(d, "*")))
                    if os.path.isdir(x) and re.search(r"seed(\d+)", os.path.basename(x))]
            if runs:
                return d, runs
    return None, []


def parse_log(path):
    """与 run_e0_baseline_v2.parse_log 同口径（含 P4b 的不锚行首修正）。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Total number of parameters: (\d+)", txt)
    n_params = int(m.group(1)) if m else None
    starts = re.findall(_TS + r" .*?Epoch=(\d+) start ", txt)
    ends = re.findall(_TS + r" .*?Epoch=(\d+) end ", txt)
    early = re.search(r"Epoch==?\d+ early stop", txt)
    from datetime import datetime
    secs = []
    prev = None
    if starts:
        prev = datetime.strptime(starts[0][0], "%Y-%m-%d %H:%M:%S,%f")
    for ts, _ in ends:
        cur = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        if prev is not None:
            secs.append((cur - prev).total_seconds())
        prev = cur
    mv = re.search(r"Validation evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    mt = re.search(r"Test evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    return {
        "n_params": n_params,
        "n_epoch_start_lines": len(starts),
        "n_epoch_end_lines": len(ends),
        "n_epochs": len(ends) + (1 if early else 0),
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
        "epoch_seconds": secs,
        "valid_auc": float(mv.group(2)) if mv else None,
        "test_auc": float(mt.group(2)) if mt else None,
        "has_traceback": "Traceback (most recent call last)" in txt,
        "log_bytes": os.path.getsize(path),
    }


# ---------------------------------------------------------------------------
# [第 5 项] parse_log 实现交叉验证
# ---------------------------------------------------------------------------
# 为什么需要：同一个「从日志解析 epoch 耗时」的逻辑被抄到了 8 个文件里。
# 实测至少两处各自独立出过 bug（都用 "Epoch=N start" 做时间差分，而 fuxictr
# 只在开头打一次 start；以及用 `^时间戳` 锚行首，而并入 stderr 后时间戳不在行首）。
# 三个人（captain / runner / infra）都在「一处改对、另一处没同步」上翻过车。
#
# 更麻烦的是：**用作者日志验证会通过，只有用我们自己产生的日志才暴露**——
# 作者日志不并流 stderr，时间戳在行首；我们的 train.log 并了 tqdm，时间戳在行中。
# 这条已写成教训。
#
# 所以不重构，改为「让不一致自动暴露」：拿同一份日志跑遍所有能找到的 parse 实现，
# 比较 (n_epochs, mean_epoch_seconds)。任何实现给出 end=0 / mean=None 就 FAIL。
# ---------------------------------------------------------------------------
def _extract_module_parser(path):
    """尝试从某个 .py 里 import 出「解析单个 run 日志」的函数。

    各文件命名不统一（实测）：
        run_e0_baseline*.py     -> parse_log
        e0_guard.py             -> parse_run_log
        split_train_eval_time   -> analyze（吃路径，返回 dict）
    所以按优先级试多个名字，不要只认一个。
    """
    import importlib.util
    name = "chk_" + re.sub(r"\W", "_", os.path.basename(path)[:-3])
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:
        return None, "import 失败: %s" % type(exc).__name__
    for fn_name in ("parse_log", "parse_train_log", "parse_run_log", "analyze"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn, None
    return None, "无已知的解析函数"


def _normalize(d):
    """把不同实现的返回统一成 (n_epochs, mean_epoch_seconds)。

    实测有两大类返回形态：
      * dict：run_e0_baseline*.py / e0_guard.py -> 有 n_epochs / mean_epoch_seconds
      * list：split_train_eval_time.analyze()   -> [(epoch_no, train_s, eval_s, total_s), ...]
    只认 dict 会把 list 型实现误判成 "解析不出来"（假阳性）。
    """
    if isinstance(d, dict):
        return d.get("n_epochs"), d.get("mean_epoch_seconds")
    if isinstance(d, (list, tuple)):
        if not d:
            return 0, None
        # 每项形如 (ep, tr, ev, tot)；取最后一列作为单轮总秒数
        tots = []
        for row in d:
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                tots.append(row[3])
            elif isinstance(row, (int, float)):
                tots.append(row)
        if tots:
            return len(tots), (sum(tots) / len(tots))
        return len(d), None
    return None, None


def cross_check_parsers(sample_log, tools_dir, quiet=False):
    """对 sample_log 跑遍 tools_dir 下所有含 parse 实现的脚本，返回 (fails, warns, rows)。

    KNOWN_DEPRECATED 里的文件若解析失败，记为 warn 而不是 FAIL：
    它们是已知的废弃实现，其 FAIL 不影响任何结论，但会稀释注意力。
    """
    fails, warns, rows = [], [], []
    if not os.path.exists(sample_log):
        return fails, warns, rows
    if not os.path.isdir(tools_dir):
        return fails, warns, rows

    cands = sorted(glob.glob(os.path.join(tools_dir, "*.py")))
    for p in cands:
        base = os.path.basename(p)
        try:
            full = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if "Epoch=" not in full:
            continue
        fn, err = _extract_module_parser(p)
        if fn is None:
            rows.append((base, "-", "-", "跳过（%s）" % err))
            continue
        try:
            n, mean = _normalize(fn(sample_log))
        except Exception as exc:
            rows.append((base, "-", "-", "调用异常 %s" % type(exc).__name__))
            continue
        deprecated = base in KNOWN_DEPRECATED
        note = "已知废弃（预期内，不影响结论）" if deprecated else ""
        rows.append((base, n, mean, note))
        txt = open(sample_log, encoding="utf-8", errors="replace").read()
        n_end = len(re.findall(_TS + r" .*?Epoch=\d+ end ", txt))
        if n_end >= 1:
            msg = None
            if mean is None:
                msg = "%s 解析出 mean_epoch_seconds=None（应为数值）" % base
            elif n is not None and n <= 1 and n_end >= 2:
                msg = "%s 解析出 n_epochs=%s，但日志有 %d 个 end 行" % (base, n, n_end)
            if msg:
                if deprecated:
                    warns.append("[已知废弃] " + msg)
                else:
                    fails.append(msg)

    if rows and not quiet:
        print("  %-34s %-9s %-12s %s" % ("实现文件", "n_epochs", "mean_s/ep", "备注"))
        for base, n, mean, note in rows:
            print("  %-34s %-9s %-12s %s"
                  % (base, n,
                     ("%.1f" % mean) if isinstance(mean, (int, float)) else mean,
                     note))
    return fails, warns, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0 开跑 5 分钟启动自检（硬断言）")
    ap.add_argument("--workdir", required=True, help="传给 driver 的 --workdir（项目根）")
    ap.add_argument("--seeds", default=None,
                    help="期望的种子列表，逗号分隔；给了就硬校验完全一致")
    ap.add_argument("--expect-params", type=int, default=EXPECT_PARAMS)
    ap.add_argument("--quiet", action="store_true", help="只输出结论与失败项")
    ap.add_argument("--skip-parser-crosscheck", action="store_true",
                    help="跳过第 5 项（parse 实现交叉验证）")
    args = ap.parse_args(argv)

    fails, warns = [], []

    def say(*a):
        if not args.quiet:
            print(*a)

    print("=" * 88)
    print("E0 启动自检  workdir=%s" % args.workdir)
    print("=" * 88)

    # ================= 1. seed_list.json =================
    say("\n[1] seed_list.json")
    found = find_seed_list(args.workdir)
    if not found:
        print("  [FAIL] 找不到 seed_list.json")
        print("         已找过：%s/runs/ 、%s/" % (args.workdir, args.workdir))
        fails.append("缺 seed_list.json（E3 无法复用种子列表 -> 配对设计会静默失效）")
    else:
        p, how = found[0]
        say("  [OK]   找到：%s" % p)
        say("         来源：%s" % how)
        if len(found) > 1:
            warns.append("发现 %d 份 seed_list.json（%s），可能有多份不一致"
                         % (len(found), [x[0] for x in found]))
        if "搜出来的" in how:
            fails.append("seed_list.json 位置非预期（%s）—— E3 按约定路径读不到" % p)
        try:
            d = json.load(open(p, encoding="utf-8"))
            seeds_got = [int(s) for s in d.get("seeds", [])]
            say("         seeds = %s" % seeds_got)
            say("         exp/lin = %s/%s, expect_params = %s"
                % (d.get("exp_layers"), d.get("lin_layers"), d.get("expect_params")))
            if args.seeds:
                want = [int(s) for s in args.seeds.split(",") if s.strip()]
                if seeds_got != want:
                    print("  [FAIL] seed 列表与命令行不一致")
                    print("         文件：%s" % seeds_got)
                    print("         期望：%s" % want)
                    fails.append("seed_list 内容与 --seeds 不一致（E3 会配错对）")
                else:
                    say("  [OK]   与前缀 --seeds 完全一致（%d 个）" % len(want))
            if d.get("expect_params") not in (None, args.expect_params):
                fails.append("seed_list 的 expect_params=%s ≠ %d"
                             % (d.get("expect_params"), args.expect_params))
        except Exception as exc:
            print("  [FAIL] 解析失败：%s" % exc)
            fails.append("seed_list.json 解析失败")

    # ================= 2. run 目录与参数量 =================
    say("\n[2] run 目录与参数量")
    base_dir, runs = find_run_dirs(args.workdir)
    if not base_dir:
        print("  [FAIL] 找不到任何 e0_baseline/<...seedN> 目录")
        fails.append("没有任何 run 目录（driver 可能还没启动，或 --workdir 给错）")
        active = None
    else:
        say("  [OK]   %s（%d 个 run）" % (base_dir, len(runs)))
        active = None
        for d in runs:
            log = os.path.join(d, "train.log")
            if not os.path.exists(log):
                say("    %-8s 无 train.log" % os.path.basename(d))
                continue
            r = parse_log(log)
            seed = os.path.basename(d)
            say("    %-26s params=%-10s start=%d end=%d epochs=%d mean=%s  %s"
                % (seed, r["n_params"], r["n_epoch_start_lines"], r["n_epoch_end_lines"],
                   r["n_epochs"],
                   ("%.1fs" % r["mean_epoch_seconds"]) if r["mean_epoch_seconds"] else "None",
                   "TRACEBACK!" if r["has_traceback"] else ""))
            if r["has_traceback"]:
                fails.append("%s 的 train.log 里有 traceback" % seed)
            if r["n_params"] is not None and r["n_params"] != args.expect_params:
                fails.append("%s 参数量 %d ≠ %d"
                             % (seed, r["n_params"], args.expect_params))
            # 记录「当前正在跑的那个」（日志非空且还没 test 结果）
            if r["log_bytes"] > 0 and r["test_auc"] is None:
                active = (seed, r, log)

    # ================= 3. 计时解析（硬断言）=================
    say("\n[3] 计时解析是否生效（P4 探针）")
    probe = None
    if active:
        probe = active[1]
        say("  用正在跑的 %s 检查（日志 %d 字节）" % (active[0], probe["log_bytes"]))
    elif runs:
        # 没有正在跑的，就用第一个已完成的
        for d in runs:
            log = os.path.join(d, "train.log")
            if os.path.exists(log):
                probe = parse_log(log)
                say("  用已完成的 %s 检查" % os.path.basename(d))
                break

    if probe is None:
        print("  [FAIL] 没有可检查的日志")
        fails.append("无日志可验计时解析")
    else:
        say("  n_epoch_start_lines = %d   （正常应为 1，fuxictr 只打一次 start）"
            % probe["n_epoch_start_lines"])
        say("  n_epoch_end_lines   = %d   （每轮一个 end；早停轮只有 early stop）"
            % probe["n_epoch_end_lines"])
        say("  n_epochs            = %d" % probe["n_epochs"])
        say("  mean_epoch_seconds  = %s" % probe["mean_epoch_seconds"])

        # --- 硬断言 ---
        # 判据要精确区分「P4 复现」与「只是才跑了 1 轮」：
        #   P4 的特征是：end 行匹配不到（end_lines==0）或 mean 算不出来（None）
        #   「刚跑完第 1 轮」时 n_epochs==1 是完全正常的，不能判 FAIL
        # （第一版把 n_epochs<=1 直接判 FAIL，在真实 E0 上误报了 —— 这是假阳性，已修）
        if probe["n_epoch_end_lines"] == 0:
            print("  [FAIL] n_epoch_end_lines = 0")
            print("         说明正则一条都没匹配到 —— P4b 回来了（时间戳不在行首）")
            print("         检查 train.log 里 Epoch 行是否被 tqdm 进度条并流到行中")
            fails.append("计时解析失效（end_lines=0）")
        elif probe["mean_epoch_seconds"] is None:
            print("  [FAIL] mean_epoch_seconds = None（有 %d 个 end 行却算不出耗时）"
                  % probe["n_epoch_end_lines"])
            fails.append("mean_epoch_seconds=None（计时不可用，报告会缺这项）")
        else:
            secs = probe["epoch_seconds"]
            say("  [OK]   计时解析生效 ✅ 单轮 %.1fs（%d 个样本：%s）"
                % (probe["mean_epoch_seconds"], len(secs),
                   ", ".join("%.1f" % s for s in secs)))
            if probe["n_epochs"] == 1:
                # 不是 FAIL，只是提示「还在第 1 轮/刚结束第 1 轮」
                print("  [info] 当前只完成 %d 轮 —— 正常（E0 才刚开始）。"
                      "跑几轮后再看 n_epochs 会增长。" % probe["n_epochs"])
            # 合理性区间：DCU 实测 228.6s/epoch（纯训练），含 valid 评估约 271s
            m = probe["mean_epoch_seconds"]
            if m < 30:
                warns.append("单轮仅 %.1fs，远低于 DCU 实测 228.6s —— "
                             "可能数据没读全或 batch 数不对" % m)
            elif m > 600:
                warns.append("单轮 %.1fs，远高于 DCU 实测 228.6s+eval42s≈271s —— "
                             "可能机器有并发负载（共享环境），此值不可作官方数字" % m)

    # ================= 4. 附加：seed_list 与 run 目录一致性 =================
    say("\n[4] seed_list 与 run 目录一致性")
    if found and runs:
        try:
            want = [int(s) for s in json.load(open(found[0][0], encoding="utf-8"))["seeds"]]
            got = sorted(int(re.search(r"seed(\d+)", os.path.basename(d)).group(1))
                         for d in runs)
            say("  seed_list: %s" % want)
            say("  已有目录: %s" % got)
            missing = [s for s in want if s not in got]
            if missing:
                say("  [info] 尚未开始（正常，driver 是串行）：%s" % missing)
            unexpected = [s for s in got if s not in want]
            if unexpected:
                print("  [FAIL] 目录里有 seed_list 之外的种子：%s" % unexpected)
                fails.append("run 目录含种子列表外的 seed（E3 配对会错）")
        except Exception:
            pass

    # ================= 5. parse 实现交叉验证 =================
    if not args.skip_parser_crosscheck:
        say("\n[5] parse_log 实现交叉验证（防「一处改对、另一处没同步」）")
        # 选日志：优先用我们自己的（含 tqdm 并流），因为它才是暴露 bug 的那个
        sample = None
        if active:
            sample = active[2]
        elif runs:
            for d in runs:
                p = os.path.join(d, "train.log")
                if os.path.exists(p):
                    sample = p
                    break
        if sample is None:
            say("  跳过：没有可用日志")
        else:
            say("  样本日志：%s" % sample)
            say("  （刻意用我们自己的日志 —— 作者日志不并流 stderr，时间戳在行首，")
            say("    用它会掩盖 `^锚行首` 这类 bug）")
            tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            cfails, cwarns, crows = cross_check_parsers(sample, tools_dir, quiet=args.quiet)
            warns.extend(cwarns)
            if cfails:
                print("  [FAIL] %d 个实现与预期不符：" % len(cfails))
                for f in cfails:
                    print("         - %s" % f)
                fails.extend(cfails)
            elif crows:
                say("  [OK]   所有在役实现结果一致，无静默失效 ✅")
            if cwarns:
                say("  [info] %d 处已知废弃实现的失败已按 warn 处理（见汇总）" % len(cwarns))

    # ================= 汇总 =================
    print("\n" + "=" * 88)
    if warns:
        print("警告（不阻塞，但需注意）：")
        for w in warns:
            print("  ! %s" % w)
    if fails:
        print("启动自检：FAIL（%d 项）" % len(fails))
        for f in fails:
            print("  x %s" % f)
        print("\n不要继续等 10 小时 —— 立刻排查。")
        print("退出码 1")
        return 1
    print("启动自检：全部通过 ✅  E0 已正常启动")
    print("  seed_list 就位 / 参数量正确 / 计时解析生效")
    print("退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
