#!/usr/bin/env python3
"""通用 CSV -> h5 转换器（支持 Criteo / iPinYou / KKBox 等任意数据集）。

## 为什么需要它
`tools/build_h5.py` 是 **Criteo 专用**的：字段列表（I1-I13 / C1-C26）、
vocab 校验值（910747）、FeatureProcessor 全部写死在代码里。

本脚本把这三样都变成参数：
  1. 字段定义  -> 从 dataset_config.yaml 读
  2. 校验值    -> 用 --expect-vocab 传（不传就只打印，不判定）
  3. 处理器    -> 按数据集名自动选择（这是官方 run_expid.py 崩溃的根因）

## FeatureProcessor 选择规则（关键）
fuxictr 2.0.1 里不同数据集有各自的 FeatureProcessor 子类：
  Criteo  -> fuxictr.datasets.criteo.FeatureProcessor   (需要 convert_to_bucket)
  KKBox   -> fuxictr.datasets.kkbox.FeatureProcessor    (需要 genre/artist 序列处理)
  其他    -> fuxictr.preprocess.FeatureProcessor        (基类，如 iPinYou)
选错会 AttributeError 或 vocab 对不上。

## CSV 目录怎么定位
配置里的路径形如 `../../../data/KKBox_x1/train.csv`，子目录（KKBox_x1）不能丢。
默认从配置里自动推导 `--data-root/<子目录>/`；也可以显式传 `--csv-dir` 覆盖。

## 用法（iPinYou）
  python3 tools/build_h5_any.py \
      --config code/FCN_repo/config/dataset_config.yaml \
      --config-key iPinYou_x1_csv \
      --data-root data/ \
      --dataset-id iPinYou_x1_h5 \
      --out data/iPinYou_x1_h5 \
      --expect-vocab 665765 --expect-fields 16

## 用法（KKBox，注意配置键名是 KKBox_csv）
  python3 tools/build_h5_any.py \
      --config code/FCN_repo/config/dataset_config.yaml \
      --config-key KKBox_csv \
      --data-root data/ \
      --dataset-id KKBox_x1_h5 \
      --out data/KKBox_x1_h5 \
      --expect-vocab 91756 --expect-fields 13 \
      --emb-dim 128

## 内存纪律
强制 data_block_size=0（单进程）。开了会 mp.Pool(cpu_count()//2)，
cgroup 只给 6 核，多进程超卖且每个子进程各持一份数据副本，内存成倍上涨。
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

# ---- FeatureProcessor 路由表：dataset 名关键字 -> (模块, 类名) ----
PROCESSOR_MAP = {
    "criteo": ("fuxictr.datasets.criteo", "FeatureProcessor"),
    "kkbox": ("fuxictr.datasets.kkbox", "FeatureProcessor"),
}
DEFAULT_PROCESSOR = ("fuxictr.preprocess", "FeatureProcessor")


def pick_processor(dataset_name):
    low = dataset_name.lower()
    for key, target in PROCESSOR_MAP.items():
        if key in low:
            return target, key
    return DEFAULT_PROCESSOR, "default"


def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="通用 CSV -> h5 转换")
    ap.add_argument("--config", required=True, help="dataset_config.yaml 路径")
    ap.add_argument("--config-key", required=True, help="配置里的数据集键名，如 iPinYou_x1_csv / KKBox_csv")
    ap.add_argument("--data-root", required=True, help="数据集根目录，如 data/")
    ap.add_argument("--csv-dir", default=None,
                    help="含 train/valid/test.csv 的目录；不传则从配置推导为 <data-root>/<子目录>/")
    ap.add_argument("--dataset-id", required=True, help="生成的 h5 数据集 id，如 iPinYou_x1_h5")
    ap.add_argument("--out", required=True, help="输出目录，通常与 --dataset-id 同名")
    ap.add_argument("--min-categr-count", type=int, default=None,
                    help="不传则用配置里的值")
    ap.add_argument("--emb-dim", type=int, default=16, help="仅用于校验时加载 FeatureMap")
    ap.add_argument("--expect-vocab", type=int, default=None, help="Σ(vocab_size) 期望值")
    ap.add_argument("--expect-fields", type=int, default=None, help="字段数期望值")
    ap.add_argument("--data-block-size", type=int, default=0)
    ap.add_argument("--keep-feature-map", action="store_true",
                    help="feature_map.json 已存在时保留（默认删除重建）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    # numpy 兼容：本机 numpy 2.x 没有 np.Inf，服务器 1.25 有，加了无害
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    # ---- 1. 读配置 ----
    cfg_all = load_yaml(args.config)
    if args.config_key not in cfg_all:
        logging.error("配置里没有键 %r，可选：%s", args.config_key, sorted(cfg_all.keys()))
        sys.exit(1)
    cfg = cfg_all[args.config_key]

    data_root = args.data_root.rstrip("/") + "/"

    # 定位 csv 目录：子目录名不能丢（KKBox 在 data/KKBox_x1/ 下）
    if args.csv_dir:
        csv_dir = args.csv_dir.rstrip("/") + "/"
        src = "--csv-dir 显式指定"
    else:
        sub = os.path.basename(os.path.dirname(cfg["train_data"].rstrip("/")))
        csv_dir = os.path.join(data_root, sub) + "/"
        src = "从配置推导"
    logging.info("CSV 目录（%s）: %s", src, csv_dir)

    def rebuild(p):
        return os.path.join(csv_dir, os.path.basename(p))

    params = {
        "data_format": "csv",
        "data_root": data_root,
        "dataset_id": args.dataset_id,
        "train_data": rebuild(cfg["train_data"]),
        "valid_data": rebuild(cfg["valid_data"]),
        "test_data": rebuild(cfg["test_data"]),
        "label_col": cfg["label_col"],
        "min_categr_count": (args.min_categr_count if args.min_categr_count is not None
                             else cfg.get("min_categr_count", 1)),
        "feature_cols": cfg["feature_cols"],
        "data_block_size": args.data_block_size,
        "verbose": 0,
    }

    # ---- 2. 挑 FeatureProcessor ----
    (mod_name, cls_name), matched = pick_processor(args.config_key + " " + args.dataset_id)
    logging.info("数据集 %s -> 使用 %s.%s", args.config_key, mod_name, cls_name)
    mod = __import__(mod_name, fromlist=[cls_name])
    FeatureProcessor = getattr(mod, cls_name)

    from fuxictr.features import FeatureMap
    from fuxictr.preprocess import build_dataset

    # ---- 3. 检查输入 ----
    for k in ("train_data", "valid_data", "test_data"):
        p = params[k]
        if not os.path.isfile(p):
            logging.error("%s 不存在：%s", k, p)
            sys.exit(1)
        logging.info("%-11s %s  (%.1f MB)", k, p, os.path.getsize(p) / 1e6)

    # ---- 4. 转换 ----
    fmj = os.path.join(args.out, "feature_map.json")
    if os.path.exists(fmj) and not args.keep_feature_map:
        logging.warning("feature_map.json 已存在，删除后重建：%s", fmj)
        os.remove(fmj)

    logging.info("开始 CSV -> h5（min_categr_count=%s, data_block_size=%d）",
                 params["min_categr_count"], params["data_block_size"])
    t0 = time.time()
    try:
        fe = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(fe, **params)
    except TypeError as e:
        logging.error("FeatureProcessor 初始化失败（很可能是选错了处理器）：%s", e)
        logging.error("当前用的是 %s.%s，调整脚本里的 PROCESSOR_MAP 再试", mod_name, cls_name)
        raise
    elapsed = (time.time() - t0) / 60
    logging.info("转换完成，耗时 %.2f 分钟", elapsed)
    gc.collect()

    # ---- 5. 校验 ----
    fm = FeatureMap(params["dataset_id"], args.out)
    fm.load(fmj, {"embedding_dim": args.emb_dim})
    got = {k: v["vocab_size"] for k, v in fm.features.items()}
    total = int(sum(got.values()))
    n_fields = int(fm.num_fields)

    print()
    print("=" * 64)
    ok_vocab = True
    if args.expect_vocab is not None:
        ok_vocab = (total == args.expect_vocab)
        print("校验 vocab : %-4s 实测 %d  期望 %d" %
              ("PASS" if ok_vocab else "FAIL", total, args.expect_vocab))
    else:
        print("校验 vocab : 未提供期望值，仅报告 Σ(vocab_size) = %d" % total)

    ok_fields = True
    if args.expect_fields is not None:
        ok_fields = (n_fields == args.expect_fields)
        print("校验字段数: %-4s 实测 %d  期望 %d" %
              ("PASS" if ok_fields else "FAIL", n_fields, args.expect_fields))
    else:
        print("校验字段数: 未提供期望值，实测 %d" % n_fields)
    print("=" * 64)

    # ---- 6. 落盘 ----
    summary = {
        "dataset_id": args.dataset_id,
        "config_key": args.config_key,
        "csv_dir": csv_dir,
        "processor": "%s.%s" % (mod_name, cls_name),
        "processor_matched": matched,
        "n_fields": n_fields,
        "vocab_sum": total,
        "expect_vocab": args.expect_vocab,
        "expect_fields": args.expect_fields,
        "vocab_ok": ok_vocab,
        "fields_ok": ok_fields,
        "min_categr_count": params["min_categr_count"],
        "data_block_size": args.data_block_size,
        "elapsed_minutes": round(elapsed, 2),
        "vocab_all": got,
    }
    with open(os.path.join(args.out, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n产物：")
    total_mb = 0.0
    for fn in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, fn)
        if os.path.isfile(p):
            mb = os.path.getsize(p) / 1e6
            total_mb += mb
            print("  %-26s %8.1f MB" % (fn, mb))
    print("  %-26s %8.1f MB" % ("[合计]", total_mb))

    if not (ok_vocab and ok_fields):
        print("\n!! 校验未通过，请勿继续实验。")
        print("   常见原因：CSV 不完整 / min_categr_count 不对 / FeatureProcessor 选错")
        sys.exit(1)
    print("\n校验通过。")


if __name__ == "__main__":
    main()
