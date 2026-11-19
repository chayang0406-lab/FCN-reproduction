#!/usr/bin/env python3
"""h5 独立验收：逐字段核对 vocab_size 是否与作者日志完全一致。

=============================================================================
为什么要有「独立复核」而不是用 data 的自检
-----------------------------------------------------------------------------
t4 的验收标准是：`feature_map.json` 里 39 个字段的 vocab_size 必须与作者日志
`FCNv2_Criteo_016_cde6d2ff.log` 的 Feature specs **逐字段相等**，合计 910,747。

这条标准比「跑出 AUC 0.8157」硬得多，因为它**不受随机种子影响**：
vocab 是数据的确定性属性，错了就是数据管道口径不对，AUC 再好看也不能用。

但如果只有 data 自己的自检，存在「自检脚本和转换脚本共享同一个 bug」的风险
（比如两者都用 min_categr_count=2 而不是 10，自检会认为一致）。
所以 runner 用**独立写的**脚本再核一次，输入只有 feature_map.json 和作者日志原文。

=============================================================================
本脚本查 6 项
-----------------------------------------------------------------------------
1. 三个 h5 文件存在且大小合理
2. feature_map.json 存在且格式正确（list-of-single-key-dict）
3. 39 个字段**逐字段** vocab_size 与作者日志相等（不是只看总和）
4. Σvocab == 910,747（论文 Table 2 的 Criteo #Features）
5. feature_map.json 的 dataset_id == Criteo_x4_10_h5
6. 由 vocab 推出的参数量 == 16,532,562（与 E0 闸门对齐）

用法：
    python3 verify_h5_dataset.py --data-root data
    python3 verify_h5_dataset.py --data-root data --strict   # 任一失败 exit 1
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys

DATASET_ID = "Criteo_x4_10_h5"

# 作者日志 checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_016_cde6d2ff.log
# 「Feature specs」段逐字段抄录（这是原文，不是我从别处推的）
AUTHOR_VOCAB = {
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
    "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
    "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
    "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
    "C25": 82, "C26": 40955,
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
    "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
}
AUTHOR_SUM = 910747
EMB_DIM = 16
D_DIM = 39 * EMB_DIM           # 624
PER_LAYER_4_1 = D_DIM * (D_DIM // 2) + (D_DIM // 2) + (D_DIM // 2) + D_DIM
EXPECT_PARAMS_4_1 = AUTHOR_SUM * EMB_DIM + 2 * (5 * PER_LAYER_4_1 + (D_DIM + 1))


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return "%.2f %s" % (n, unit)
        n /= 1024.0
    return "%.2f PB" % n


def main(argv=None):
    ap = argparse.ArgumentParser(description="h5 独立验收")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--strict", action="store_true", help="有 FAIL 时 exit 1")
    ap.add_argument("--vocab-only", action="store_true",
                    help="只验 feature_map.json 的 vocab（valid/test h5 尚未生成时用）。"
                         "此时 h5 缺失降级为 info，不记为 FAIL")
    args = ap.parse_args(argv)

    ddir = os.path.join(args.data_root, args.dataset_id)
    fails, warns = [], []

    print("=" * 84)
    print("h5 独立验收（runner 写的第三方复核，不复用 t4 的自检代码）")
    print("数据集目录：%s" % ddir)
    print("=" * 84)

    # ---------------- 1. 三个 h5 ----------------
    print("\n[1] h5 文件")
    missing_h5 = []
    for split in ("train", "valid", "test"):
        p = os.path.join(ddir, split + ".h5")
        if not os.path.exists(p):
            if args.vocab_only:
                # --vocab-only 模式下，h5 缺失是正常的：feature_map.json 在
                # fit(train) + transform_h5(train) 之后就已落盘，而那时 valid/test
                # 还没转。此时 vocab 已经确定，可以提前验收（比等三份都转完早约 1 小时）。
                print("  [info] 缺 %s（--vocab-only 模式不检查）" % os.path.basename(p))
                missing_h5.append(split)
            else:
                print("  [FAIL] 缺 %s" % p)
                fails.append("缺 %s.h5" % split)
        else:
            sz = os.path.getsize(p)
            flag = "OK" if sz > 1024 else "FAIL(过小)"
            print("  [%s] %-10s %s" % (flag, split + ".h5", human(sz)))
            if sz <= 1024:
                fails.append("%s.h5 过小" % split)

    # ---------------- 2. feature_map.json ----------------
    print("\n[2] feature_map.json")
    fm_path = os.path.join(ddir, "feature_map.json")
    if not os.path.exists(fm_path):
        print("  [FAIL] 缺 %s" % fm_path)
        fails.append("缺 feature_map.json")
        fm = None
    else:
        try:
            fm = json.load(open(fm_path, encoding="utf-8"))
            print("  [OK]   已读到（%s）" % human(os.path.getsize(fm_path)))
        except Exception as exc:
            print("  [FAIL] 解析失败：%s" % exc)
            fails.append("feature_map.json 解析失败")
            fm = None

    if fm is not None:
        # 格式检查：必须是 list-of-single-key-dict
        feats = fm.get("features")
        bad_fmt = []
        if not isinstance(feats, list):
            bad_fmt.append("features 不是 list")
        else:
            for i, item in enumerate(feats):
                if not isinstance(item, dict) or len(item) != 1:
                    bad_fmt.append("第 %d 项不是单键 dict（fuxictr 2.0.1 要求）" % i)
                    break
        if bad_fmt:
            print("  [FAIL] 格式错误：%s" % "; ".join(bad_fmt))
            fails.append("feature_map.json 格式错误")
        else:
            print("  [OK]   格式正确（%d 项，每项单键 dict）" % len(feats))

        # dataset_id
        got_id = fm.get("dataset_id")
        if got_id != args.dataset_id:
            print("  [FAIL] dataset_id=%r，应为 %r" % (got_id, args.dataset_id))
            fails.append("dataset_id 不匹配")
        else:
            print("  [OK]   dataset_id = %s" % got_id)

        # ---------------- 3. 逐字段 vocab ----------------
        print("\n[3] 39 字段逐字段 vocab_size 与作者日志比对")
        got = {}
        if isinstance(feats, list):
            for item in feats:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, dict) and "vocab_size" in v:
                            got[k] = int(v["vocab_size"])
        print("  作者日志 %d 个字段 / 本地 %d 个字段" % (len(AUTHOR_VOCAB), len(got)))
        missing = sorted(set(AUTHOR_VOCAB) - set(got))
        extra = sorted(set(got) - set(AUTHOR_VOCAB))
        if missing:
            print("  [FAIL] 本地缺字段：%s" % missing)
            fails.append("缺字段 %s" % missing)
        if extra:
            print("  [FAIL] 本地多出字段：%s" % extra)
            fails.append("多出字段 %s" % extra)
        diffs = []
        for k in sorted(AUTHOR_VOCAB):
            if k in got and got[k] != AUTHOR_VOCAB[k]:
                diffs.append((k, AUTHOR_VOCAB[k], got[k]))
        if diffs:
            print("  [FAIL] %d 个字段 vocab_size 不符：" % len(diffs))
            print("         %-6s %-10s %-10s %s" % ("字段", "作者", "本地", "差"))
            for k, a, b in diffs:
                print("         %-6s %-10d %-10d %+d" % (k, a, b, b - a))
            fails.append("%d 个字段 vocab 不符" % len(diffs))
        elif not missing and not extra:
            print("  [OK]   39/39 字段**逐字段完全一致** ✅")

        # ---------------- 4. Σvocab ----------------
        print("\n[4] Σ(vocab_size)")
        s = sum(got.values())
        if s == AUTHOR_SUM:
            print("  [OK]   %d == 作者日志 910,747 ✅（也等于论文 Table 2 的 Criteo #Features）" % s)
        else:
            print("  [FAIL] %d ≠ 910,747（差 %+d）" % (s, s - AUTHOR_SUM))
            fails.append("Σvocab=%d ≠ 910747" % s)

        # ---------------- 5. 由 vocab 推参数量 ----------------
        print("\n[5] 由 vocab 推出的 E0 参数量（应与闸门 16,532,562 一致）")
        n = s * EMB_DIM + 2 * (5 * PER_LAYER_4_1 + (D_DIM + 1))
        if n == EXPECT_PARAMS_4_1:
            print("  [OK]   %d == 16,532,562 ✅" % n)
        else:
            print("  [FAIL] %d ≠ 16,532,562（差 %+d）" % (n, n - EXPECT_PARAMS_4_1))
            fails.append("参数量 %d ≠ 16532562" % n)

        # num_fields
        nf = fm.get("num_fields")
        if nf != 39:
            print("  [FAIL] num_fields=%r，应为 39" % nf)
            fails.append("num_fields≠39")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 84)
    if fails:
        print("验收结果：FAIL（%d 项）" % len(fails))
        for f in fails:
            print("  - %s" % f)
        print("\n!! 不要用这份 h5 开跑。E0 的参数量闸门会报 MISMATCH，")
        print("   但更早发现更省时间（h5 转换要 2-3 小时，E0 要 11 小时）。")
    elif args.vocab_only and missing_h5:
        print("验收结果：vocab 全部通过 ✅（但 %s 的 h5 尚未生成）"
              % "/".join(missing_h5))
        print("  这在 --vocab-only 模式下是正常的：feature_map.json 在 train 转完后")
        print("  就已落盘，vocab 已经确定，可提前验收。")
        print("  三份 h5 齐备后再跑一次不带 --vocab-only 的完整验收。")
    else:
        print("验收结果：全部通过 ✅  可以开跑 E0")
        print("  建议命令：见 docs/开跑前路径错位检查_20261118.md §五")
    print("=" * 84)

    if args.strict and fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
