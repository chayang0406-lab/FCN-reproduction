#!/usr/bin/env python3
"""构造一个「vocab 规模与真实 Criteo 完全一致、但行数很少」的 h5 数据集。

用途：在真实数据（4584 万行、h5 转换要几小时）就绪之前，把 E0 的**整条链路**验证到
参数量闸门真正放行——因为参数量只取决于 vocab 规模，不取决于行数：

    params = 910,747 x 16 + 2 x [(4+1) x 195,936 + 625] = 16,532,562

所以只要 39 个字段的 vocab_size 与作者日志逐字段相等，模型就会是 16,532,562 个参数，
闸门就会打印 OK。这能证明「配置生效、feature_map 口径对、闸门真的会放行」，
而不是只会报 MISMATCH。

⚠️ 本数据集**只用于管道验证**：行数是造的、标签是随机的，AUC 无任何参考意义。
   严禁把它的任何指标写进报告。

用法：
  python3 make_synthetic_full_vocab.py --out-dir <dir> [--rows 20000] [--seed 7]
"""
import argparse
import json
import os

import h5py
import numpy as np

# 作者日志 checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_016_cde6d2ff.log 里
# Feature specs 段的逐字段 vocab_size（合计 910,747 = 论文 Table 2 的 Criteo #Features）
C_VOCAB = {
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
    "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
    "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
    "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
    "C25": 82, "C26": 40955,
}
I_VOCAB = {
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
    "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
}
ORDER = (["I%d" % i for i in range(1, 14)] +
         ["C%d" % i for i in range(1, 27)])   # 与 dataset_config feature_cols 顺序一致

DATASET_ID = "Criteo_x4_10_h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    vocab = dict(I_VOCAB)
    vocab.update(C_VOCAB)
    assert sum(vocab.values()) == 910747, "vocab 总和应为 910,747，实际 %d" % sum(vocab.values())
    assert len(vocab) == 39, "字段数应为 39，实际 %d" % len(vocab)

    data_dir = os.path.join(args.out_dir, DATASET_ID)
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    def write_split(name, rows):
        path = os.path.join(data_dir, name + ".h5")
        if os.path.exists(path):
            os.remove(path)
        with h5py.File(path, "w") as h5:
            for f in ORDER:
                # padding_idx=0 保留，取值 1..vocab_size-1，模拟真实 ID 分布
                h5.create_dataset(f, data=rng.randint(1, vocab[f], size=rows,
                                                     dtype=np.int64))
            # 标签随机：本数据只验证管道，不用于任何性能结论
            h5.create_dataset("Label", data=rng.randint(0, 2, size=rows).astype(np.float32))
        return path

    print("生成 %s  (%d 行/份, 39 字段, Σvocab=%d)"
          % (data_dir, args.rows, sum(vocab.values())))
    for split, rows in (("train", args.rows), ("valid", max(args.rows // 10, 128)),
                        ("test", max(args.rows // 10, 128))):
        p = write_split(split, rows)
        print("  %-6s %8d 行  %s" % (split, rows, os.path.basename(p)))

    # feature_map.json：格式必须是「每项只有一个键的 dict」组成的 list
    # （fuxictr 2.0.1 的 FeatureMap.load 用 for x in features for k,v in x.items()）
    features = []
    for f in ORDER:
        v = vocab[f]
        features.append({f: {"source": "", "type": "categorical",
                             "padding_idx": 0, "oov_idx": v - 1, "vocab_size": v}})
    fm = {"dataset_id": DATASET_ID, "num_fields": len(ORDER),
          "total_features": sum(vocab.values()), "labels": ["Label"],
          "features": features}
    fm_path = os.path.join(data_dir, "feature_map.json")
    with open(fm_path, "w", encoding="utf-8") as fp:
        json.dump(fm, fp, indent=2)
    print("  feature_map.json  Σvocab=%d  n_fields=%d"
          % (fm["total_features"], fm["num_fields"]))

    # 自查：参数量必须等于 16,532,562
    emb = sum(vocab.values()) * 16
    per_layer = 624 * 312 + 312 + 312 + 624
    expect = emb + 2 * (5 * per_layer + 625)
    print("\n预期参数量 = %d  (%s)"
          % (expect, "OK == 16,532,562" if expect == 16532562 else "!! 不等于 16,532,562"))


if __name__ == "__main__":
    main()
