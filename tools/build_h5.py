#!/usr/bin/env python3
"""把 Criteo_x4 的 CSV 转成 fuxictr 需要的 h5，并做双硬校验。

## 为什么要自己写而不直接用 run_expid.py
官方 `run_expid.py` 有已知缺陷（见 docs/官方脚本的三个缺陷_20261118.md）：
- 导入了 fuxictr 2.0.1 不存在的 `delete_model_files`
- FeatureProcessor 被 KKBox 版覆盖（Criteo 需要 convert_to_bucket）
- dataset_config 缺 Criteo_x4_10_h5 定义

## 内存与并发纪律（见 docs/CSV转h5内存实测_20261118.md）
实测 4584 万行一次性读入峰值约 43.7 GB（上限 115 GB，占 38%）。
- 强制 `data_block_size=0`：不要开 mp.Pool(cpu_count()//2)，
  因为宿主报 32 核但 cgroup 只给 6 核，多进程会超卖 + 内存成倍
- 转完主动释放 object DataFrame，避免与 int 版叠加

## 双硬校验（都不依赖随机性）
1. Σ(vocab_size) == 910,747，且逐字段与作者日志一致
2. 用 exp=4/lin=1 建模型，参数量 == 16,532,562
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

# 作者日志里的真实 vocab_size（逐字抄自 FCNv2_Criteo.log 的 Feature specs）
AUTHOR_VOCAB = {
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
    "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
    "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
    "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
    "C25": 82, "C26": 40955,
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
    "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
}
AUTHOR_VOCAB_SUM = 910747
EXPECT_PARAMS = 16532562


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="含 train/valid/test.csv 的目录")
    ap.add_argument("--out", required=True, help="输出 Criteo_x4_10_h5 目录")
    ap.add_argument("--min-categr-count", type=int, default=10)
    ap.add_argument("--data-block-size", type=int, default=0,
                    help="必须为 0（多进程会超卖且内存成倍）")
    ap.add_argument("--repo", default=None, help="FCN 仓库，用于参数量校验")
    ap.add_argument("--skip-param-check", action="store_true")
    ap.add_argument("--allow-subset", action="store_true",
                    help="仅用于小样本试跑：vocab/参数校验不通过也不退出（不写正式结论）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    # numpy 兼容（本机 numpy 2.x 需要，服务器 1.25 不需要但无害）
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    from fuxictr.features import FeatureMap
    from fuxictr.preprocess import build_dataset
    from fuxictr.datasets.criteo import FeatureProcessor   # 关键：必须是 criteo 版

    params = {
        "data_format": "csv",
        "data_root": os.path.dirname(args.out.rstrip("/")) + "/",
        "dataset_id": os.path.basename(args.out.rstrip("/")),
        "train_data": os.path.join(args.csv_dir, "train.csv"),
        "valid_data": os.path.join(args.csv_dir, "valid.csv"),
        "test_data": os.path.join(args.csv_dir, "test.csv"),
        "label_col": {"dtype": "float", "name": "Label"},
        "min_categr_count": args.min_categr_count,
        "feature_cols": [
            {"active": True, "dtype": "float", "fill_na": 0, "na_value": 0,
             "name": ["I%d" % i for i in range(1, 14)],
             "preprocess": "convert_to_bucket", "type": "categorical"},
            {"active": True, "dtype": "str", "fill_na": "", "na_value": "",
             "name": ["C%d" % i for i in range(1, 27)], "type": "categorical"},
        ],
        "data_block_size": args.data_block_size,
        "verbose": 0,
    }

    fmj = os.path.join(args.out, "feature_map.json")
    if os.path.exists(fmj):
        logging.warning("feature_map.json 已存在，删除后重建：%s" % fmj)
        os.remove(fmj)

    logging.info("开始 CSV -> h5（data_block_size=%d）" % args.data_block_size)
    t0 = time.time()
    fe = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(fe, **params)
    logging.info("转换完成，耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    gc.collect()

    # ---- 校验 1：vocab ----
    # 注意：FeatureMap.load 的 params 里**必须有 embedding_dim**，
    # 否则 default_emb_dim 为 None，后面 sum_emb_out_dim() 会
    # TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'
    fm = FeatureMap(params["dataset_id"], args.out)
    fm.load(fmj, {"embedding_dim": 16})
    got = {k: v["vocab_size"] for k, v in fm.features.items()}
    total = sum(got.values())
    logging.info("字段数 = %d，Σ(vocab_size) = %d" % (fm.num_fields, total))

    mismatches = []
    for k in sorted(AUTHOR_VOCAB):
        if k not in got:
            mismatches.append((k, AUTHOR_VOCAB[k], None))
        elif got[k] != AUTHOR_VOCAB[k]:
            mismatches.append((k, AUTHOR_VOCAB[k], got[k]))

    ok_vocab = (total == AUTHOR_VOCAB_SUM) and not mismatches
    print()
    print("=" * 64)
    if args.allow_subset and not ok_vocab:
        print("校验 1（vocab）: SKIPPED-BY-FLAG（--allow-subset，仅小样本试跑用）")
        print("  Σ(vocab_size) = %d（全量应为 %d）" % (total, AUTHOR_VOCAB_SUM))
    else:
        print("校验 1（vocab）: %s" % ("PASS" if ok_vocab else "FAIL"))
    print("  Σ(vocab_size) = %d ，期望 %d" % (total, AUTHOR_VOCAB_SUM))
    if mismatches:
        print("  不一致字段（共 %d 个，列前 15）：" % len(mismatches))
        for k, exp, act in mismatches[:15]:
            print("    %-5s 期望 %-8s 实际 %s" % (k, exp, act))
    print("=" * 64)

    # ---- 校验 2：参数量 ----
    ok_param = None
    if not args.skip_param_check and args.repo:
        sys.path.insert(0, args.repo)
        import src as model_zoo
        mp = {
            "model_root": args.out, "verbose": 0, "task": "binary_classification",
            "gpu": -1, "monitor": {"AUC": 1, "logloss": 0}, "monitor_mode": "max",
            "early_stop_patience": 2, "eval_steps": None, "save_best_only": True,
            "embedding_regularizer": 1e-5, "net_regularizer": 0,
            "optimizer": "adam", "loss": "binary_crossentropy",
            "metrics": ["logloss", "AUC"], "embedding_dim": 16,
            "exp_num_layers": 4, "lin_num_layers": 1, "net_dropout": 0.1,
            "batch_norm": False, "layer_norm": False, "num_heads": 1,
            "model_id": "pcheck",
        }
        m = getattr(model_zoo, "FCNv2")(fm, **mp)
        n = sum(p.numel() for p in m.parameters())
        ok_param = (n == EXPECT_PARAMS)
        print("校验 2（参数量 exp=4/lin=1）: %s" % ("PASS" if ok_param else "FAIL"))
        print("  实测 %d ，期望 %d" % (n, EXPECT_PARAMS))
        print("=" * 64)

    # ---- 落盘总结 ----
    summary = {
        "dataset_dir": args.out,
        "n_fields": int(fm.num_fields),
        "vocab_sum": int(total),
        "vocab_sum_expected": AUTHOR_VOCAB_SUM,
        "vocab_ok": bool(ok_vocab),
        "vocab_mismatches": [{"field": k, "expected": e, "actual": a}
                             for k, e, a in mismatches],
        "params_ok": ok_param,
        "params_expected": EXPECT_PARAMS,
        "min_categr_count": args.min_categr_count,
        "data_block_size": args.data_block_size,
        "elapsed_minutes": (time.time() - t0) / 60,
        "vocab_all": got,
    }
    with open(os.path.join(args.out, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 列出产物
    print("\n产物：")
    for fn in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, fn)
        if os.path.isfile(p):
            print("  %-26s %.1f MB" % (fn, os.path.getsize(p) / 1e6))

    if not ok_vocab and not args.allow_subset:
        print("\n!! vocab 校验失败，请勿继续实验。检查 min_categr_count 与 CSV 是否完整。")
        sys.exit(1)
    if ok_param is False and not args.allow_subset:
        print("\n!! 参数量校验失败，请勿继续实验。")
        sys.exit(1)
    if args.allow_subset:
        print("\n[--allow-subset] 小样本试跑：校验结果不作数，禁止用于正式结论。")
        return
    print("\n全部校验通过，可以开始实验。")


if __name__ == "__main__":
    main()

