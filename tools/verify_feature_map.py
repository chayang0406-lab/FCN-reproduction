#!/usr/bin/env python3
"""独立校验 Criteo_x4_10_h5 的 feature_map.json 是否与作者日志同口径。

这个脚本与 tools/build_h5.py 的校验**互相独立**：
build_h5.py 是转换器自带的（自己建、自己验），本脚本是**只读**的第三方核对，
可以单独对已有产物运行，用在「产物已经生成好了，我来验一遍」的场景。

## 两个硬断言（都不依赖随机性）
1. Σ(vocab_size) == 910,747
   —— 等于论文 Table 2 的 Criteo #Features
2. 39 个字段的 vocab_size 逐一等于作者日志 FCNv2_Criteo.log 的值
   —— 这比总和相等更严：总和相等也可能是各字段偏差互相抵消

## 附带的诊断信号
- 若 I 字段 vocab 明显偏小（如 I1 ≈ 11-19 而非 44），说明 bucketize 用了 log2 而非 (ln x)^2
- 若某 C 字段大幅偏离，先查 min_categr_count 是否为 10

用法:
    python3 verify_feature_map.py --h5-dir <Criteo_x4_10_h5 目录>
    python3 verify_feature_map.py --h5-dir ... --json out.json
退出码: 0 = PASS, 1 = FAIL
"""

import argparse
import json
import os
import sys

# 作者日志 FCNv2_Criteo.log 的 Feature specs，逐字抄录（勿改）
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

# 论文 Table 2 的 Criteo #Features，作为第二来源
PAPER_TABLE2_FEATURES = 910747

# 期望的 h5 行数
EXPECT_ROWS = {"train": 36672493, "valid": 4584062, "test": 4584062}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-h5", action="store_true")
    a = ap.parse_args()

    fmj = os.path.join(a.h5_dir, "feature_map.json")
    if not os.path.exists(fmj):
        print(f"FAIL: 找不到 {fmj}")
        return 1

    with open(fmj, encoding="utf-8") as f:
        fm = json.load(f)

    # fuxictr 2.0.1 的 feature_map.json 结构：
    #   {"dataset_id":..., "num_fields":39, "total_features":910747,
    #    "labels":[...], "features":[ {"I1": {...}}, {"I2": {...}}, ... ]}
    # 注意 features 是 **list of 单键 dict**，不是 dict。早期版本是 dict，
    # 所以这里两种都兼容，避免换 fuxictr 版本后脚本失效。
    raw_feats = fm.get("features", {})
    got = {}
    if isinstance(raw_feats, dict):
        items = raw_feats.items()
        for name, spec in items:
            if isinstance(spec, dict) and "vocab_size" in spec:
                got[name] = int(spec["vocab_size"])
    elif isinstance(raw_feats, list):
        for entry in raw_feats:
            if not isinstance(entry, dict):
                continue
            for name, spec in entry.items():
                if isinstance(spec, dict) and "vocab_size" in spec:
                    got[name] = int(spec["vocab_size"])
    else:
        print(f"FAIL: 无法识别的 features 类型 {type(raw_feats).__name__}")
        return 1

    total = sum(got.values())
    out = {
        "feature_map": fmj,
        "n_fields": len(got),
        "vocab_sum": total,
        "vocab_sum_expected": AUTHOR_SUM,
        "paper_table2_features": PAPER_TABLE2_FEATURES,
        "num_fields_in_json": fm.get("num_fields"),
        "total_features_in_json": fm.get("total_features"),
    }

    print("=" * 68)
    print(f"字段数 = {len(got)}（期望 39）；json 内 num_fields = {fm.get('num_fields')}")
    print(f"Σ(vocab_size) = {total:,}（期望 {AUTHOR_SUM:,}）；"
          f"json 内 total_features = {fm.get('total_features')}")
    print("=" * 68)

    # 断言 1
    ok_sum = (total == AUTHOR_SUM)
    print(f"断言 1（Σvocab == 910,747，= 论文 Table 2 #Features）: {'PASS' if ok_sum else 'FAIL'}")

    # 断言 2：逐字段
    mismatches = []
    missing = []
    for k in sorted(AUTHOR_VOCAB):
        if k not in got:
            missing.append(k)
        elif got[k] != AUTHOR_VOCAB[k]:
            mismatches.append({"field": k, "expected": AUTHOR_VOCAB[k], "actual": got[k]})
    extra = [k for k in got if k not in AUTHOR_VOCAB]

    ok_fields = not mismatches and not missing and not extra
    print(f"断言 2（39 字段逐一相等）: {'PASS' if ok_fields else 'FAIL'}")
    if mismatches:
        print(f"  不一致字段 {len(mismatches)} 个：")
        for m in mismatches[:20]:
            d = m["actual"] - m["expected"]
            print(f"    {m['field']:<5} 期望 {m['expected']:<8} 实际 {m['actual']:<8} 差 {d:+d}")
    if missing:
        print(f"  缺失字段: {missing}")
    if extra:
        print(f"  多余字段: {extra}")

    out["vocab_sum_ok"] = ok_sum
    out["fields_ok"] = ok_fields
    out["mismatches"] = mismatches
    out["missing"] = missing
    out["extra"] = extra
    out["vocab_all"] = dict(sorted(got.items()))

    # ---- 诊断：bucketize 规则是否用错 ----
    print()
    print("--- 诊断：I 字段 bucketize 规则 ---")
    i1 = got.get("I1")
    if i1 is not None:
        if i1 >= 40:
            print(f"  I1 vocab = {i1}（作者 44）→ 与 ⌊(ln x)²⌋ 一致 ✅")
        elif i1 <= 20:
            print(f"  I1 vocab = {i1} → ⚠️ 明显偏小！疑似用了 ⌊log2 x⌋（上界仅 19）。")
            print(f"     作者实现是 int(np.floor(np.log(x) ** 2))，即 ⌊(ln x)²⌋。")
            out["bucketize_suspect_log2"] = True
        else:
            print(f"  I1 vocab = {i1} → 介于两者之间，需人工判断")
    out["I1_vocab"] = i1

    # ---- h5 行数校验 ----
    if not a.skip_h5:
        print()
        print("--- h5 行数与体积 ---")
        rows_ok = True
        h5info = {}
        try:
            import h5py
            for split, exp in EXPECT_ROWS.items():
                p = os.path.join(a.h5_dir, f"{split}.h5")
                if not os.path.exists(p):
                    print(f"  {split}.h5 缺失 ⚠️")
                    h5info[split] = {"error": "missing"}
                    rows_ok = False
                    continue
                with h5py.File(p, "r") as hf:
                    keys = list(hf.keys())
                    # 行数取任一非标量数据集的第 0 维（各字段行数必然相同）
                    n = None
                    for k in keys:
                        if hf[k].ndim >= 2 - 1 and hf[k].shape:
                            n = int(hf[k].shape[0])
                            break
                    # 优先用 label 列（fuxictr 里叫 "Label"，大写 L）
                    for cand in ("Label", "label", "labels"):
                        if cand in hf:
                            n = int(hf[cand].shape[0])
                            break
                    dtypes = sorted({str(hf[k].dtype) for k in keys})
                sz = os.path.getsize(p)
                ok = (n == exp)
                rows_ok = rows_ok and ok
                nstr = f"{n:,}" if n is not None else "None"
                print(f"  {split}.h5 rows={nstr} (期望 {exp:,}) {'OK' if ok else 'FAIL'}  "
                      f"{sz/1e9:.3f} GB  n_datasets={len(keys)}  dtypes={dtypes}")
                h5info[split] = {"rows": n, "expected": exp, "rows_ok": ok,
                                 "bytes": sz, "n_datasets": len(keys),
                                 "dtypes": dtypes}
        except ImportError:
            print("  (h5py 不可用，跳过 h5 校验)")
        out["h5"] = h5info
        out["h5_rows_ok"] = rows_ok
    else:
        rows_ok = True

    verdict = ok_sum and ok_fields and rows_ok
    out["verdict"] = "PASS" if verdict else "FAIL"

    print()
    print("=" * 68)
    print(f"总判定: {out['verdict']}")
    print("=" * 68)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入 {a.json}")

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
