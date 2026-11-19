#!/usr/bin/env python3
"""E3/E4 配对分析：把「改进 vs 基线」的多种子结果做成可写进报告的统计表。

=============================================================================
为什么必须单独写一个：配对设计目前「无工具强制执行」
-----------------------------------------------------------------------------
项目纪律写的是「E3 必须复用 E0 的同一份种子列表（配对设计），
能把需要的样本量从 16 降到 8」。但实测发现：
  * `seed_list.json` 只被**写**，没有任何脚本**读**它；
  * `run_improved.py` 只有 `--seed`（单值），没有 `--seeds`（从文件读）；
  * 没有任何配对统计工具。
也就是说这条纪律**全靠人记**。一旦 E3 用了不同的种子（哪怕多一个少一个），
配对就退化成独立样本，需要的样本量翻倍，而报告里看不出来。

本工具做两件事：
  1. **硬校验配对性**：两组的种子集合必须**完全相同**，否则拒绝出统计（exit 2）。
  2. **配对检验**：逐 seed 求差 → 报告 ΔAUC 均值、95% CI、配对 t 检验与
     Wilcoxon 符号秩检验，并按实测 σ_Δ 反算「本次设计能分辨多大效应」。

=============================================================================
统计口径说明（写给非统计背景的读者）
-----------------------------------------------------------------------------
* **为什么要配对**：同一台机器上，种子 2025 恰好让模型收敛得好一点、
  种子 2027 恰好差一点，这个「种子运气」对基线和改进是**共通的**。
  直接比两组均值，这个运气会变成噪声；先逐 seed 相减，运气就抵消了。
  类比：让同一批学生做两套卷子，比「A 班平均分 vs B 班平均分」更靠谱——
  因为同一个学生的水平差异被消掉了。
* **σ_Δ 与 σ 的区别**：σ 是「同一配置不同种子」的波动（噪声基线，E0 测出来的），
  σ_Δ 是「配对差值的波动」。若种子效应真的共通，σ_Δ 会远小于 σ。
  **σ_Δ 才是决定「能不能测出改进」的量**，不是 σ。
* **δ_min（本次实验的分辨率）**：用实测 σ_Δ 和实际配对数 n 反算出
  「以 80% 把握能测出的最小效应」。若 |ΔAUC| < δ_min，只能写
  「未观察到可分辨差异」，**不能写「无效」**——absence of evidence ≠ evidence of absence。

用法：
  # 两组的 h5/目录名不同时用 --baseline-dir / --improved-dir
  python3 paired_analysis.py \
      --baseline-dir runs/e0_baseline \
      --improved-dir runs/e3_soft \
      --seed-list runs/seed_list.json \
      --expect-params 16532562 \
      --out runs/e3_paired.json

  # 目录里每个 seed 一个子目录，子目录含 train.log 或 result.json
  # （两种来源都支持：E0 是 train.log+正则解析，E3 是 result.json）
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys


# ============================ 日志解析（与 E0 一致）============================
_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)"


def parse_train_log(path):
    """从 fuxictr 日志解析指标。与 run_e0_baseline_v2.parse_log 保持同口径。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Total number of parameters: (\d+)", txt)
    n_params = int(m.group(1)) if m else None
    mv = re.search(r"Validation evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    mt = re.search(r"Test evaluation.*?Metrics\] logloss: ([\d.]+) - AUC: ([\d.]+)",
                   txt, re.S)
    # 兜底：run_improved.py 的日志里没有 "Validation evaluation" 字样，
    # 但有 "valid={...}" 这一行；result.json 更可靠，优先用它（见 load_run）。
    valid = (float(mv.group(2)), float(mv.group(1))) if mv else (None, None)
    test = (float(mt.group(2)), float(mt.group(1))) if mt else (None, None)
    return {"n_params": n_params,
            "valid_auc": valid[0], "valid_logloss": valid[1],
            "test_auc": test[0], "test_logloss": test[1]}


def load_run(run_dir):
    """读一个 run 的结果。优先 result.json（全精度），退回 train.log（6 位小数）。"""
    rj = os.path.join(run_dir, "result.json")
    if os.path.exists(rj):
        try:
            r = json.load(open(rj, encoding="utf-8"))
            return {"source": "result.json",
                    "n_params": r.get("n_params_total") or r.get("n_params"),
                    "valid_auc": (r.get("valid") or {}).get("AUC"),
                    "valid_logloss": (r.get("valid") or {}).get("logloss"),
                    "test_auc": (r.get("test") or {}).get("AUC"),
                    "test_logloss": (r.get("test") or {}).get("logloss")}
        except Exception:
            pass
    tl = os.path.join(run_dir, "train.log")
    if os.path.exists(tl):
        d = parse_train_log(tl)
        d["source"] = "train.log（6 位小数）"
        return d
    return None


_SEED_RE = re.compile(r"seed(\d+)")


def discover_runs(parent, seed_list):
    """在 parent 下找每个 seed 的 run 目录，返回 {seed: dir}。

    目录名里的 seed 用 `seed(\d+)` 抓（E0 是 e0_exp4_lin1_seed2025，
    E3 可以叫 e3_soft_seed2025，都能匹配）。
    也支持 parent 本身就是单个 run 目录（只有一层）。
    """
    if not os.path.isdir(parent):
        return {}
    found = {}
    # 情况 1：parent 下每个子目录一个 run
    for d in sorted(glob.glob(os.path.join(parent, "*"))):
        if not os.path.isdir(d):
            continue
        m = _SEED_RE.search(os.path.basename(d))
        if not m:
            continue
        s = int(m.group(1))
        if seed_list and s not in seed_list:
            continue
        found[s] = d
    # 情况 2：parent 下还有一层 runs/e0_baseline/seedN
    if not found:
        for d in sorted(glob.glob(os.path.join(parent, "*", "*"))):
            if not os.path.isdir(d):
                continue
            m = _SEED_RE.search(os.path.basename(d))
            if not m:
                continue
            s = int(m.group(1))
            if seed_list and s not in seed_list:
                continue
            found[s] = d
    return found


# ============================ 统计 ============================
def t_critical_975(df):
    """双侧 α=0.05 的 t 临界值（小样本查表，df 1..30，超出用 1.96）。"""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
             19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
             25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    return table.get(int(df), 1.96)


def paired_t(diffs):
    """配对 t 检验。返回 (t, p_two_sided, df)。"""
    n = len(diffs)
    if n < 2:
        return None, None, max(n - 1, 0)
    md = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return None, None, n - 1
    t = md / (sd / math.sqrt(n))
    df = n - 1
    # p 值用 t 分布近似：|t| 与临界值比较给出区间，再给一个粗估
    tc = t_critical_975(df)
    if abs(t) >= tc:
        p_approx = 0.01 if abs(t) > tc * 1.6 else 0.03
    else:
        p_approx = 0.10 if abs(t) > tc * 0.7 else 0.50
    return t, p_approx, df


def wilcoxon_signrank(diffs):
    """Wilcoxon 符号秩检验（双侧，小样本用正态近似）。返回 (W, p_approx, n_used)。"""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 1:
        return None, None, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    W = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, None, n
    z = (W - mu) / sigma
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return W, p_approx, n


def delta_min(sigma_delta, n, alpha=0.05, power=0.80):
    """本次设计（配对数 n、差值标准差 σ_Δ）能以 power 把握测出的最小效应。

    非中心 t 的近似：(t_{1-α/2} + t_{power}) × σ_Δ / sqrt(n)
    → (1.96 + 0.8416) × σ_Δ / sqrt(n) = 2.8016 × σ_Δ / sqrt(n)
    """
    if n < 2 or sigma_delta is None:
        return None
    return (1.959964 + 0.841621) * sigma_delta / math.sqrt(n)


def n_needed(sigma_delta, delta, alpha=0.05, power=0.80):
    """要分辨 delta 需要多少配对数。"""
    if sigma_delta is None or delta == 0:
        return None
    k = 1.959964 + 0.841621
    return (k * sigma_delta / delta) ** 2


# ============================ 主流程 ============================
def collect(parent, seed_list, expect_params, label):
    runs = discover_runs(parent, seed_list)
    out = {}
    print("  [%s] 在 %s 下找到 %d 个 run" % (label, parent, len(runs)))
    for s in sorted(runs):
        r = load_run(runs[s])
        if r is None:
            print("    seed %-5d !! 目录里既无 result.json 也无 train.log：%s"
                  % (s, runs[s]))
            continue
        bad = []
        if r.get("n_params") is None:
            bad.append("无参数量")
        elif expect_params and r["n_params"] != expect_params:
            bad.append("参数量 %s ≠ %s" % (r["n_params"], expect_params))
        if r.get("valid_auc") is None or r.get("test_auc") is None:
            bad.append("缺 valid/test AUC")
        if bad:
            print("    seed %-5d EXCLUDED：%s" % (s, "; ".join(bad)))
            continue
        r["seed"] = s
        r["dir"] = runs[s]
        out[s] = r
        print("    seed %-5d valid=%.6f test=%.6f params=%s  (%s)"
              % (s, r["valid_auc"], r["test_auc"], r["n_params"], r["source"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3/E4 配对分析")
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--improved-dir", required=True)
    ap.add_argument("--seed-list", default=None,
                    help="seed_list.json 路径；给了就硬校验两组种子集合相同")
    ap.add_argument("--expect-params", type=int, default=None,
                    help="参数量闸门；不符的 run 直接排除，不参与统计")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--label", default="improved", help="改进组名称，仅用于输出")
    args = ap.parse_args(argv)

    seeds = None
    if args.seed_list and os.path.exists(args.seed_list):
        j = json.load(open(args.seed_list, encoding="utf-8"))
        seeds = sorted(int(s) for s in j["seeds"])
        print("种子列表（来自 %s）：%s" % (args.seed_list, seeds))
    else:
        print("种子列表：未提供（将取两组的交集，可能掩盖配对性破坏）")

    print("\n读取基线组：")
    base = collect(args.baseline_dir, seeds, args.expect_params, "baseline")
    print("\n读取改进组：")
    impr = collect(args.improved_dir, seeds, args.expect_params, args.label)

    # ---------------- 配对性硬校验 ----------------
    print("\n" + "=" * 78)
    if seeds:
        missing_b = [s for s in seeds if s not in base]
        missing_i = [s for s in seeds if s not in impr]
        if missing_b or missing_i:
            print("!! 配对性破坏：seed_list 声明了 %d 个种子，但有效结果不全" % len(seeds))
            if missing_b:
                print("   基线组缺：%s" % missing_b)
            if missing_i:
                print("   改进组缺：%s" % missing_i)
            print("\n配对设计的前提是「同一个种子两组都有」。缺一个就无法相减，")
            print("退化成独立样本后，所需样本量约翻倍（16 → 8 的前提不再成立）。")
            print("请补跑缺失种子，或用 --seed-list 指定一个更小的公共子集（并记录原因）。")
            return 2
        common = seeds
    else:
        common = sorted(set(base) & set(impr))
        if set(base) != set(impr):
            print("!! 警告：两组种子集合不同（基线 %d 个 / 改进 %d 个），"
                  "取交集 %d 个" % (len(base), len(impr), len(common)))
            print("   这会降低统计功效，且报告里必须写明剔除了哪些种子。")
    if len(common) < 2:
        print("!! 配对数 %d < 2，无法做统计" % len(common))
        return 3

    print("配对种子（%d 个）：%s" % (len(common), common))

    # ---------------- 逐 seed 求差 ----------------
    rows = []
    for s in common:
        b, i = base[s], impr[s]
        rows.append({
            "seed": s,
            "base_valid": b["valid_auc"], "impr_valid": i["valid_auc"],
            "d_valid": i["valid_auc"] - b["valid_auc"],
            "base_test": b["test_auc"], "impr_test": i["test_auc"],
            "d_test": i["test_auc"] - b["test_auc"],
            "base_valid_ll": b["valid_logloss"], "impr_valid_ll": i["valid_logloss"],
            "d_valid_ll": (i["valid_logloss"] - b["valid_logloss"])
                           if (b["valid_logloss"] is not None
                               and i["valid_logloss"] is not None) else None,
        })

    d_valid = [r["d_valid"] for r in rows]
    d_test = [r["d_test"] for r in rows]

    def summarize(diffs, name, sign_hint):
        n = len(diffs)
        md = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else None
        se = sd / math.sqrt(n) if sd is not None else None
        tc = t_critical_975(n - 1)
        ci = (md - tc * se, md + tc * se) if se is not None else (None, None)
        t, p_t, df = paired_t(diffs)
        W, p_w, n_w = wilcoxon_signrank(diffs)
        dmin = delta_min(sd, n)
        wins = sum(1 for x in diffs if x > 0)
        print("\n--- %s（配对，n=%d）---" % (name, n))
        print("  ΔAUC 均值      : %+.6f" % md)
        if sd is not None:
            print("  σ_Δ（差值标准差）: %.6f   ← 决定能否测出改进的量" % sd)
            print("  95%% 置信区间    : [%+.6f, %+.6f]" % ci)
        print("  配对 t 检验    : t=%s, p≈%s, df=%s"
              % ("%.3f" % t if t is not None else "NA", p_t, df))
        print("  Wilcoxon 符号秩 : W=%s, p≈%s" % (W, "%.4f" % p_w if p_w else "NA"))
        print("  改进赢的种子数  : %d/%d" % (wins, n))
        if dmin is not None:
            print("  本次分辨率 δ_min: %.6f  → 小于它的差异本实验测不出" % dmin)
            verdict = ("达到可分辨" if abs(md) >= dmin else "落在分辨率之下")
            print("  判定（%s）: %s" % (sign_hint, verdict))
        ns = n_needed(sd, 0.001) if sd is not None else None
        if ns:
            # 向上取整；不足 1 个时明确写「<1」，不要印成「0 个」（会看着像不需要样本）
            shown = "<1" if ns < 1 else "%d" % math.ceil(ns)
            print("  要分辨 Δ=0.001 需配对 %s 个（本次 %d 个）" % (shown, n))
        return {"n": n, "mean_delta": md, "sigma_delta": sd,
                "ci95": list(ci) if ci[0] is not None else None,
                "t": t, "p_t_approx": p_t, "df": df,
                "wilcoxon_W": W, "wilcoxon_p_approx": p_w,
                "wins": wins, "delta_min": dmin,
                "n_needed_for_0.001": ns}

    s_valid = summarize(d_valid, "valid AUC（早停选模所用，报告主线用这个）", "valid")
    s_test = summarize(d_test, "test AUC（早停后评估，系统性偏高，仅作对照）", "test")

    # ---------------- 结论文字 ----------------
    print("\n" + "=" * 78)
    print("可直接写进报告的结论（注意区分 valid / test）")
    print("=" * 78)
    if s_valid["delta_min"] is not None:
        md, dmin = s_valid["mean_delta"], s_valid["delta_min"]
        if abs(md) >= dmin:
            print("  valid：ΔAUC = %+.6f，达到本次实验分辨率（δ_min=%.6f），"
                  "方向为%s。" % (md, dmin, "提升" if md > 0 else "下降"))
        else:
            print("  valid：ΔAUC = %+.6f，**未观察到可分辨差异**"
                  "（本次 δ_min=%.6f）。" % (md, dmin))
            print("  注意：这不等于「证明无效」，而是「本实验条件下分辨率不足」。")
            print("  学术表述：absence of evidence ≠ evidence of absence。")
    print("  必须写明：配对数 n=%d、σ_Δ=%.6f、α=0.05、power=0.80。"
          % (s_valid["n"], s_valid["sigma_delta"] or -1))
    if s_valid["mean_delta"] * s_test["mean_delta"] < 0:
        print("  ⚠️ valid 与 test 的差**符号相反**，不能合并成一个结论，必须分列报告。")

    out = {
        "baseline_dir": args.baseline_dir,
        "improved_dir": args.improved_dir,
        "improved_label": args.label,
        "seed_list_file": args.seed_list,
        "seeds_paired": common,
        "expect_params": args.expect_params,
        "rows": rows,
        "valid": s_valid,
        "test": s_test,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("\n已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
