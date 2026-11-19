#!/usr/bin/env python3
"""改进实验（E3/E4-A/E4-B）开跑前的多维度闸门。

=============================================================================
为什么需要这个：参数量**不足以**作为改进实验的闸门
-----------------------------------------------------------------------------
实测 15 个 expid 的参数量，发现 8 个配置**完全相同**（16,532,562）：

    FCNv2_C_base                    4/1   16,532,562   ← 基线
    FCNv2_C_soft_smoothmax          4/1   16,532,562   ← 只改 loss 权重方式
    FCNv2_C_soft_softplus           4/1   16,532,562
    FCNv2_C_soft_ratio              4/1   16,532,562
    FCNv2_C_soft_smoothmax_anneal   4/1   16,532,562
    FCNv2_C_abl_no_tribce           4/1   16,532,562   ← 拆掉 Tri-BCE
    FCNv2_C_abl_exp0                0/5   16,532,562   ← 层数和也是 5！
    FCNv2_C_abl_lin0                5/0   16,532,562   ← 层数和也是 5！

两个后果：
 1. **E3（Tri-BCE 软权重）**：软权重不改变参数量，所以「参数量对不对」**根本无法**
    证明改进开关生效。若 `tri_bce_mode` 因拼写错误没被读进去，参数量照样是
    16,532,562，闸门照样放行，跑 8×76 分钟得到的其实是基线重复。
 2. **E4-A（消融）**：`exp0` 与 `lin0` 把层数挪到另一个子网，层数和不变，
    参数量也不变。若配置写错（比如 exp0 写成 4/1），闸门同样放行，
    「w/o ECN」这个消融就名不副实。

所以本校验器查 **4 个维度**：
    A. 参数量           — 能区分 7 个 LCA 档位
    B. (exp,lin) 层数结构 — 能区分 base / exp0 / lin0
    C. tri_bce_mode 是否被模型真正采用（读模型实例属性，不是读配置）
    D. lca_compress_ratio 是否被模型真正采用，且与参数量自洽

用法：
    # 逐配置体检（不训练，只建模，约 20 秒/配置）
    python3 verify_improved_configs.py --configs code/configs --data-root data

    # 只查指定 expid
    python3 verify_improved_configs.py --configs code/configs --data-root data \
        --expids FCNv2_C_base,FCNv2_C_abl_exp0

    # 生成闸门基准表，供后续 run 比对
    python3 verify_improved_configs.py --configs code/configs --data-root data \
        --emit-baseline runs/expected_params.json
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# 闸门基准表：从「配置应当产生什么」推出来的期望值。
# 这些数字不是从实测反推的，而是按模型结构算出来的，所以能真正抓到配置错误。
# ---------------------------------------------------------------------------
# 顶会口径：D = 39 字段 × embedding_dim 16 = 624
D_DIM = 39 * 16
# embedding 参数 = Σvocab × emb_dim，Criteo 的 Σvocab = 910,747
EMB = 910747 * 16

EXPECTED = {
    # ---- E0 基线 & E3 软权重（结构不变，参数量必然相同）----
    "FCNv2_C_base":                  dict(exp=4, lin=1, tri="hard",        lca=0.5),
    "FCNv2_C_soft_smoothmax":        dict(exp=4, lin=1, tri="smoothmax",   lca=0.5),
    "FCNv2_C_soft_softplus":         dict(exp=4, lin=1, tri="softplus",    lca=0.5),
    "FCNv2_C_soft_ratio":            dict(exp=4, lin=1, tri="ratio",       lca=0.5),
    "FCNv2_C_soft_smoothmax_anneal": dict(exp=4, lin=1, tri="smoothmax",   lca=0.5),
    # ---- E4-A 消融：层数和都是 5，但分布不同 ----
    "FCNv2_C_abl_no_tribce":         dict(exp=4, lin=1, tri="plain",       lca=0.5),
    "FCNv2_C_abl_exp0":              dict(exp=0, lin=5, tri="hard",        lca=0.5),
    "FCNv2_C_abl_lin0":              dict(exp=5, lin=0, tri="hard",        lca=0.5),
    # ---- E4-B 5 层 LCA 扫描 ----
    "FCNv2_C_r025":                  dict(exp=4, lin=1, tri="hard",        lca=0.25),
    "FCNv2_C_r075":                  dict(exp=4, lin=1, tri="hard",        lca=0.75),
    "FCNv2_C_r100":                  dict(exp=4, lin=1, tri="hard",        lca=1.0),
    # ---- E4-B 9 层 LCA 扫描（对齐论文 Table 5 的 FCN𝑝）----
    "FCNv2_C_9L_r025":               dict(exp=4, lin=5, tri="hard",        lca=0.25),
    "FCNv2_C_9L_r050":               dict(exp=4, lin=5, tri="hard",        lca=0.5),
    "FCNv2_C_9L_r075":               dict(exp=4, lin=5, tri="hard",        lca=0.75),
    "FCNv2_C_9L_r100":               dict(exp=4, lin=5, tri="hard",        lca=1.0),
}


def expected_params(exp, lin, lca):
    """按结构算交叉网络参数，加上 embedding，得总参数。

    r=1.0 走 full_rank 路径：没有 γ/β，所以每层少 2×proj 个参数。
    这一点已实测对齐（r=1 比线性外推少 11,232 = 2×9×624）。
    """
    layers = exp + lin
    if abs(lca - 1.0) < 1e-12:
        proj = D_DIM
        per_layer = D_DIM * proj + D_DIM          # w + b，无 γ/β
    else:
        proj = D_DIM // 2 if abs(lca - 0.5) < 1e-12 else max(1, int(round(D_DIM * lca)))
        per_layer = D_DIM * proj + proj + proj + D_DIM   # w + γ + β + b
    per_net = layers * per_layer + (D_DIM + 1)
    return EMB + 2 * per_net


def inspect_one(expid, cfg_dir, data_root):
    """建出模型，读出它**实际**采用了什么配置（而不是配置文件里写了什么）。"""
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    from fuxictr.utils import load_config
    from fuxictr.features import FeatureMap
    import src as zoo

    params = load_config(cfg_dir + "/", expid)
    params["gpu"] = -1                       # 只建模，不碰 GPU
    dataset_id = params["dataset_id"]
    data_dir = os.path.join(data_root, dataset_id)
    fm_path = os.path.join(data_dir, "feature_map.json")
    if not os.path.exists(fm_path):
        raise IOError("找不到 %s，无法核对 vocab 口径" % fm_path)
    fm = FeatureMap(dataset_id, data_dir)
    fm.load(fm_path, params)

    model = getattr(zoo, params["model"])(fm, **params)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_sum = int(sum(s["vocab_size"] for s in fm.features.values()))

    # ---- 读模型实例上的真实开关值（关键：不是读 params）----
    got = {
        "n_params": n_params,
        "vocab_sum": vocab_sum,
        "n_fields": int(fm.num_fields),
        # 模型对象上的属性，才代表「真正生效」
        "tri_bce_mode": getattr(model, "tri_bce_mode", "<无此属性>"),
        "lca_compress_ratio": getattr(model, "lca_compress_ratio", "<无此属性>"),
        "cfg_exp": params.get("exp_num_layers"),
        "cfg_lin": params.get("lin_num_layers"),
    }
    # 交叉网络里实际建了几个层模块（再独立核一次结构）
    for attr in ("E2LCN", "L2ECN"):
        net = getattr(model, attr, None)
        if net is not None:
            got["%s_layers" % attr] = len(getattr(net, "w", []))
    del model
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", required=True, help="配置目录（含 model_config.yaml）")
    ap.add_argument("--data-root", required=True, help="h5 所在根（含 feature_map.json）")
    ap.add_argument("--expids", default=None, help="逗号分隔；默认查全部 EXPECTED")
    ap.add_argument("--emit-baseline", default=None, help="把闸门基准表写成 JSON")
    args = ap.parse_args(argv)

    # 让 `import src` 落在项目的 code/ 下。
    # --configs 通常是 <project>/code/configs，所以要取它的**父目录** <project>/code，
    # 因为 src/ 在 code/ 下面（code/src/）。只取一层，别多取。
    code_dir = os.path.dirname(os.path.abspath(args.configs.rstrip("/")))
    if not os.path.isdir(os.path.join(code_dir, "src")):
        sys.exit("!! %s 下没有 src/ 目录，--configs 应指向 <project>/code/configs" % code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    expids = ([e.strip() for e in args.expids.split(",")] if args.expids
              else list(EXPECTED))

    print("=" * 108)
    print("改进实验配置体检（4 维：参数量 / 层数结构 / tri_bce_mode / lca_ratio）")
    print("=" * 108)
    print("%-32s %-11s %-8s %-10s %-7s %s"
          % ("expid", "n_params", "exp/lin", "tri_mode", "lca", "判定"))

    results, failures, baseline = {}, [], {}
    for e in expids:
        want = EXPECTED.get(e)
        try:
            got = inspect_one(e, args.configs, args.data_root)
        except Exception as exc:
            print("%-32s !! 建模失败: %s: %s" % (e, type(exc).__name__, str(exc)[:40]))
            failures.append((e, "建模失败 %s" % type(exc).__name__))
            continue

        problems = []
        if want is None:
            problems.append("基准表里没有该 expid，无法判定")
        else:
            w_params = expected_params(want["exp"], want["lin"], want["lca"])
            if got["n_params"] != w_params:
                problems.append("参数量 %d ≠ 期望 %d" % (got["n_params"], w_params))
            if (got["cfg_exp"], got["cfg_lin"]) != (want["exp"], want["lin"]):
                problems.append("层数 %s/%s ≠ 期望 %s/%s"
                                % (got["cfg_exp"], got["cfg_lin"], want["exp"], want["lin"]))
            # 结构独立核对：E2LCN 的层模块数应等于 exp+lin
            if "E2LCN_layers" in got and got["E2LCN_layers"] != want["exp"] + want["lin"]:
                problems.append("E2LCN 实际 %d 层 ≠ 期望 %d 层"
                                % (got["E2LCN_layers"], want["exp"] + want["lin"]))
            if got["tri_bce_mode"] != want["tri"]:
                problems.append("tri_bce_mode=%r ≠ 期望 %r"
                                % (got["tri_bce_mode"], want["tri"]))
            if not isinstance(got["lca_compress_ratio"], float) or \
                    abs(got["lca_compress_ratio"] - want["lca"]) > 1e-9:
                problems.append("lca_ratio=%r ≠ 期望 %r"
                                % (got["lca_compress_ratio"], want["lca"]))
            if got["vocab_sum"] != 910747:
                problems.append("Σvocab=%d ≠ 910,747（数据口径不对）" % got["vocab_sum"])

        status = "OK" if not problems else "FAIL"
        print("%-32s %-11d %-8s %-10s %-7s %s"
              % (e, got["n_params"], "%s/%s" % (got["cfg_exp"], got["cfg_lin"]),
                 got["tri_bce_mode"], got["lca_compress_ratio"], status))
        for p in problems:
            print("      └─ %s" % p)
        if problems:
            failures.append((e, "; ".join(problems)))

        results[e] = got
        if want is not None:
            # 闸门基准：以**期望值**为准，不以实测为准（否则配置错了也会被写进基准）
            baseline[e] = {
                "n_params": expected_params(want["exp"], want["lin"], want["lca"]),
                "exp_num_layers": want["exp"],
                "lin_num_layers": want["lin"],
                "tri_bce_mode": want["tri"],
                "lca_compress_ratio": want["lca"],
            }

    print("=" * 108)
    if failures:
        print("结果：%d/%d 个配置未通过" % (len(failures), len(expids)))
        for e, why in failures:
            print("  - %-30s %s" % (e, why))
        print("\n!! 有 FAIL，修好再跑。用错配置跑 76 分钟 × N 的代价比这 20 秒高得多。")
    else:
        print("结果：全部 %d 个配置通过，可以开跑" % len(expids))

    # ---- 参数量盲区提示 ----
    from collections import defaultdict
    groups = defaultdict(list)
    for e, b in baseline.items():
        groups[b["n_params"]].append(e)
    blind = {v: ks for v, ks in groups.items() if len(ks) > 1}
    if blind:
        print("\n提示：以下配置参数量相同，**不能只靠参数量判断配置是否生效**，"
              "本工具已用层数/tri_mode/lca 三个额外维度补上：")
        for v, ks in blind.items():
            print("  %d：%s" % (v, ", ".join(ks)))

    if args.emit_baseline:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_baseline)), exist_ok=True)
        with open(args.emit_baseline, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        print("\n闸门基准表已写入 %s（%d 个 expid）" % (args.emit_baseline, len(baseline)))

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
