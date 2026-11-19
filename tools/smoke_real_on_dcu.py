#!/usr/bin/env python3
"""在 DCU 上用**真实 Criteo 数据子集**跑通 CSV→h5→训练，并测真实 s/epoch。

与 bench_dcu_throughput.py 的区别（很重要）：
- bench 用**合成数据**测吞吐，只用于排期粗估
- 本脚本用**真实数据格式**（真实 I/C 字段分布），能暴露 bucketize、
  min_categr_count、OOV 合并等真实路径，并且能测出含 validation eval 的完整 epoch 时间

这才是 runner 排期真正该用的数字。

用法：
  python3 smoke_real_on_dcu.py --repo <FCN> --workdir <dir> --data-dir <含csv的目录>
"""
import argparse
import json
import os
import re
import sys
import time
import logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--data-dir", required=True, help="含 train/valid/test.csv 的目录")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-categr-count", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, args.repo)

    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    import torch
    import yaml
    from fuxictr.utils import load_config, set_logger
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.pytorch.dataloaders import H5DataLoader
    from fuxictr.preprocess import build_dataset
    # 【关键】必须用 criteo 版，否则 convert_to_bucket 缺失
    from fuxictr.datasets.criteo import FeatureProcessor

    import src as model_zoo

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("[env] torch %s | device %s | cuda %s" %
          (torch.__version__, device_name, torch.cuda.is_available()), flush=True)

    work = args.workdir
    os.makedirs(work, exist_ok=True)
    dataset_id = "Criteo_subset"
    data_root = os.path.join(work, "data") + "/"
    ddir = os.path.join(data_root, dataset_id)
    os.makedirs(ddir, exist_ok=True)

    print("[1/5] 写 config（真实 Criteo 字段定义，min_categr_count=%d）"
          % args.min_categr_count, flush=True)
    cfg_dir = os.path.join(work, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    ds = {
        dataset_id: {
            "data_format": "csv",
            "data_root": data_root,
            "feature_cols": [
                {"active": True, "dtype": "float", "fill_na": 0, "na_value": 0,
                 "name": ["I%d" % i for i in range(1, 14)],
                 "preprocess": "convert_to_bucket", "type": "categorical"},
                {"active": True, "dtype": "str", "fill_na": "", "na_value": "",
                 "name": ["C%d" % i for i in range(1, 27)], "type": "categorical"},
            ],
            "label_col": {"dtype": "float", "name": "Label"},
            "min_categr_count": args.min_categr_count,
            "train_data": os.path.join(args.data_dir, "train.csv"),
            "valid_data": os.path.join(args.data_dir, "valid.csv"),
            "test_data": os.path.join(args.data_dir, "test.csv"),
        }
    }
    with open(os.path.join(cfg_dir, "dataset_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(ds, f, allow_unicode=True)

    mc = {
        "model_root": os.path.join(work, "checkpoints") + "/",
        "num_workers": 4, "verbose": 1, "early_stop_patience": 2,
        "pickle_feature_encoder": True, "save_best_only": True,
        "eval_steps": None, "debug_mode": False, "group_id": None,
        "use_features": None, "feature_specs": None, "feature_config": None,
        "model": "FCNv2", "dataset_id": dataset_id,
        "loss": "binary_crossentropy", "metrics": ["logloss", "AUC"],
        "task": "binary_classification", "optimizer": "adam",
        "learning_rate": 1e-3, "embedding_regularizer": 1e-5, "net_regularizer": 0,
        "batch_size": args.batch_size, "embedding_dim": 16,
        "exp_num_layers": 4, "lin_num_layers": 1,
        "net_dropout": 0.1, "batch_norm": False, "layer_norm": False, "num_heads": 1,
        "epochs": args.epochs, "shuffle": True, "seed": 2025,
        "monitor": {"AUC": 1, "logloss": 0}, "monitor_mode": "max", "gpu": 0,
    }
    with open(os.path.join(cfg_dir, "model_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"FCNv2_subset": mc}, f, allow_unicode=True)

    print("[2/5] CSV -> h5（真实 bucketize 路径）", flush=True)
    params = load_config(cfg_dir + "/", "FCNv2_subset")
    params["gpu"] = 0
    set_logger(params)
    seed_everything(seed=params["seed"])

    fmj = os.path.join(ddir, "feature_map.json")
    if os.path.exists(fmj):
        os.remove(fmj)
    t0 = time.time()
    fe = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(fe, **params)
    print("      h5 转换耗时 %.1f s" % (time.time() - t0), flush=True)
    fm = FeatureMap(dataset_id, ddir)
    fm.load(fmj, params)

    tot_vocab = sum(s["vocab_size"] for s in fm.features.values())
    print("[3/5] feature_map: %d 字段, Σvocab=%d" % (fm.num_fields, tot_vocab), flush=True)
    print("      前5字段 vocab: %s" % {k: v["vocab_size"]
                                       for k, v in list(fm.features.items())[:5]}, flush=True)

    print("[4/5] 建模 + 参数量核对", flush=True)
    model = getattr(model_zoo, "FCNv2")(fm, **params)
    n_params = sum(p.numel() for p in model.parameters())
    D = fm.num_fields * params["embedding_dim"]
    per_layer = D * (D // 2) + (D // 2) + (D // 2) + D
    expect = tot_vocab * params["embedding_dim"] + \
        2 * ((params["exp_num_layers"] + params["lin_num_layers"]) * per_layer + D + 1)
    print("      n_params=%d  公式=%d  一致=%s" %
          (n_params, expect, n_params == expect), flush=True)
    print("      device=%s" % model.device, flush=True)

    print("[5/5] 训练 %d 个 epoch，测真实 s/epoch（含 validation eval）" % args.epochs,
          flush=True)
    tr, va = H5DataLoader(fm, stage="train", **params).make_iterator()
    t_train = time.time()
    # 注意：params 里已经有 epochs 键，不能再显式传 epochs=，否则会重复赋值
    model.fit(tr, validation_data=va, **params)
    train_total = time.time() - t_train

    res = model.evaluate(va)

    # 从日志时间戳还原每个 epoch 的耗时（最可靠）
    logf = os.path.join(work, "checkpoints", dataset_id, "FCNv2_subset.log")
    epoch_secs, n_epochs = [], 0
    if os.path.exists(logf):
        txt = open(logf, encoding="utf-8", errors="replace").read()
        from datetime import datetime
        # 【坑1】fuxictr 只在第一个 epoch 打 "Epoch=1 start"，后续 epoch 只打 "... end"。
        # 所以不能靠 start 差分，必须用：epoch1 = end1 - start1，之后 epochN = endN - end(N-1)。
        # 【坑2】不要用 ^ 锚行首：若把 stderr（tqdm 进度条）并入了 train.log，
        # 进度条不换行，时间戳会被挤到行中间，锚 ^ 会匹配不到 → 静默得到 end=0。
        # 这个 bug 极难发现，因为用**作者的**日志验证会通过（他未并流），
        # 只有用**我们自己产生的**日志才暴露。runner 实测发现，已去掉锚点。
        starts = re.findall(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Epoch=(\d+) start", txt)
        ends = re.findall(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Epoch=(\d+) end", txt)
        st = {int(e): datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f") for t, e in starts}
        en = {int(e): datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f") for t, e in ends}
        n_epochs = len(en)
        prev = st.get(1)
        for i in sorted(en):
            if prev is not None and en[i] >= prev:
                epoch_secs.append((en[i] - prev).total_seconds())
            prev = en[i]

    n_train_rows = 200000  # 子集实际行数由外部传入更准，这里从 h5 读
    try:
        import h5py
        with h5py.File(params["train_data"], "r") as h:
            n_train_rows = h["Label"].shape[0]
    except Exception:
        pass

    out = {
        "device": device_name,
        "n_fields": fm.num_fields,
        "total_vocab": tot_vocab,
        "n_params": n_params,
        "formula_expect": expect,
        "param_match": n_params == expect,
        "n_train_rows": int(n_train_rows),
        "batch_size": args.batch_size,
        "n_epochs_run": n_epochs,
        "epoch_seconds": epoch_secs,
        "mean_epoch_seconds": (sum(epoch_secs) / len(epoch_secs)) if epoch_secs else None,
        "train_total_seconds": train_total,
        "valid_result": {k: float(v) for k, v in res.items()} if hasattr(res, "items")
                        else str(res),
    }
    with open(os.path.join(work, "smoke_real_result.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n================ RESULT ================")
    print("device            : %s" % device_name)
    print("n_fields / Σvocab : %d / %d" % (fm.num_fields, tot_vocab))
    print("n_params          : %d (公式 %d, match=%s)" % (n_params, expect, n_params == expect))
    print("train rows        : %d" % n_train_rows)
    print("epochs run        : %d" % n_epochs)
    if epoch_secs:
        print("epoch seconds     : %s" % ["%.1f" % s for s in epoch_secs])
        print("mean s/epoch      : %.1f  (含 validation eval)" % (sum(epoch_secs) / len(epoch_secs)))
        # 外推到真实 36,672,493 行
        scale = 36672493 / max(n_train_rows, 1)
        est = (sum(epoch_secs) / len(epoch_secs)) * scale
        print("外推真实全量      : %.0f s/epoch  (%.2f min)  [×%.1f]" %
              (est, est / 60, scale))
        print("  对照 4090(含eval): 203.8 s  -> 倍数 %.2fx" % (est / 203.8))
        print("  20 epoch/种子     : %.1f min" % (est * 20 / 60))
        print("  7 种子            : %.1f h" % (est * 20 * 7 / 3600))
    print("valid result      : %s" % out["valid_result"])
    print("========================================")
    print("\nwrote " + os.path.join(work, "smoke_real_result.json"))


if __name__ == "__main__":
    main()

