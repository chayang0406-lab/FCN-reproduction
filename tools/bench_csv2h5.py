"""测量真实 CSV -> h5 转换的耗时与内存，用于全量预算规划。

为什么要多档而不是单点外推：
  单点外推在 I/O 与内存压力变化时会严重失真。本脚本在 25k/50k/100k/200k 四个规模上
  实测，拟合出「固定开销 + 每行边际成本」，并记录峰值 RSS，再据此给乐观/悲观区间。

同时回答一个关键问题：pandas 一次性读入 4584 万行会不会爆 115 GiB。
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import yaml

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
SUB = os.path.join(REMOTE, "data/_subset_for_dcu/train.csv")
WORK = os.path.join(REMOTE, "bench/csv2h5")


def make_subset(n_rows, out_path):
    """生成 n_rows 行的 CSV。

    优先从 20 万行子集里截取；不够则循环复制子集内容（保持字段结构一致）。
    这样可以在没有全量数据时就测到 100 万 / 200 万规模，
    大幅降低外推误差 —— 这是本脚本的核心目的。
    """
    if os.path.exists(out_path) and _count(out_path) >= n_rows:
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(SUB, "r", encoding="utf-8", errors="replace") as fi:
        header = fi.readline()
        body = fi.readlines()

    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        fo.write(header)
        while written < n_rows:
            take = min(len(body), n_rows - written)
            fo.writelines(body[:take])
            written += take
    return out_path


def _count(p):
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
    return n


CHILD = r'''
import os, sys, time, json, resource
sys.path.insert(0, "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo")
import yaml
from fuxictr.utils import load_config
from fuxictr.preprocess import build_dataset
from fuxictr.datasets.criteo import FeatureProcessor

csv_path, cfg_root, n_rows, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

params = load_config(cfg_root + "/", "BudgetProbe")
# 每个规模用独立 dataset_id，否则 fuxictr 检测到已有 feature_map.json 会直接报错退出
params["dataset_id"] = "BudgetProbe_" + tag
params["train_data"] = csv_path
params["valid_data"] = csv_path
params["test_data"] = csv_path
params["data_root"] = os.path.dirname(os.path.dirname(csv_path)) + "/"

t0 = time.time()
enc = FeatureProcessor(**params)
tr, va, te = build_dataset(enc, **params)
dt = time.time() - t0

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
h5_size = os.path.getsize(tr)
print(json.dumps({"n_rows": n_rows, "seconds": dt, "peak_rss_mb": peak_mb,
                  "h5_bytes": h5_size}))

for p in (tr, va, te):
    try: os.remove(p)
    except OSError: pass
'''

FEATURES = ["I%d" % i for i in range(1, 14)]
CATS = ["C%d" % i for i in range(1, 27)]


def write_cfg(root):
    os.makedirs(root, exist_ok=True)
    dcfg = {
        "dataset_id": "BudgetProbe",
        "data_format": "csv",
        "data_root": "",
        "feature_cols": [
            {"active": True, "dtype": "float", "fill_na": 0, "na_value": 0,
             "name": FEATURES, "preprocess": "convert_to_bucket", "type": "categorical"},
            {"active": True, "dtype": "str", "fill_na": "", "na_value": "",
             "name": CATS, "type": "categorical"},
        ],
        "label_col": {"dtype": "float", "name": "Label"},
        "min_categr_count": 10,
        "train_data": "", "valid_data": "", "test_data": "",
    }
    mcfg = {
        "model_root": os.path.join(root, "ckpt") + "/", "num_workers": 0, "verbose": 0,
        "early_stop_patience": 2, "pickle_feature_encoder": True, "save_best_only": True,
        "eval_steps": None, "debug_mode": False, "group_id": None, "use_features": None,
        "feature_specs": None, "feature_config": None, "model": "FCNv2",
        "dataset_id": "BudgetProbe", "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"], "task": "binary_classification",
        "optimizer": "adam", "learning_rate": 1e-3, "embedding_regularizer": 1e-5,
        "net_regularizer": 0, "batch_size": 8192, "embedding_dim": 16,
        "exp_num_layers": 4, "lin_num_layers": 1, "net_dropout": 0.1,
        "batch_norm": False, "layer_norm": False, "num_heads": 1, "epochs": 1,
        "shuffle": True, "seed": 2025, "monitor": {"AUC": 1, "logloss": 0},
        "monitor_mode": "max", "gpu": -1,
    }
    with open(os.path.join(root, "dataset_config.yaml"), "w") as f:
        yaml.safe_dump({"BudgetProbe": dcfg}, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(root, "model_config.yaml"), "w") as f:
        yaml.safe_dump({"BudgetProbe": mcfg}, f, sort_keys=False, allow_unicode=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    cfg_root = os.path.join(WORK, "config")
    write_cfg(cfg_root)
    child_py = os.path.join(WORK, "_child.py")
    with open(child_py, "w", encoding="utf-8") as f:
        f.write(CHILD)

    print("%-10s %-12s %-14s %-12s %s" % ("n_rows", "seconds", "us/row", "peakRSS(GB)", "h5(MB)"))
    pts = []
    for n in (200000, 500000, 1000000, 2000000):
        p = make_subset(n, os.path.join(WORK, "sub_%d.csv" % n))
        # 注意：CSV 传进去会被特征编码器读，h5 也写在同一目录
        out = subprocess.run([sys.executable, child_py, p, cfg_root, str(n), str(n)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print("  n=%d 失败:" % n, out.stderr[-600:])
            continue
        d = json.loads(out.stdout.strip().splitlines()[-1])
        pts.append(d)
        print("%-10d %-12.2f %-14.1f %-12.2f %.1f"
              % (d["n_rows"], d["seconds"], d["seconds"] / d["n_rows"] * 1e6,
                 d["peak_rss_mb"] / 1024, d["h5_bytes"] / 1024 ** 2))

    if len(pts) < 2:
        print("有效点不足")
        return 1

    xs = [p["n_rows"] for p in pts]
    ys = [p["seconds"] for p in pts]
    # 两点法（最小与最大）
    b = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a = ys[0] - b * xs[0]
    # 内存：拟合 peak_rss vs n
    ms = [p["peak_rss_mb"] for p in pts]
    mb = (ms[-1] - ms[0]) / (xs[-1] - xs[0])
    ma = ms[0] - mb * xs[0]

    print()
    print("=== 拟合 ===")
    print("  时间: t(n) = %.2f s + %.3f us/row × n" % (a, b * 1e6))
    print("  内存: RSS(n) = %.0f MB + %.3f KB/row × n" % (ma, mb))

    REAL = 45840617   # 全量三份合计（36.7M+4.58M+4.58M）；按单份最大 36.67M 算也给出
    TRAIN = 36672493
    for label, n in (("全量三份合计", REAL), ("单份最大(train)", TRAIN)):
        t = a + b * n
        rss = ma + mb * n
        print()
        print("  --- 外推 %s (%d 行) ---" % (label, n))
        print("    耗时 = %.1f s = %.2f min = %.2f h" % (t, t / 60, t / 3600))
        print("    峰值RSS = %.1f GB (%.1f GiB)" % (rss / 1024, rss / 1024 / 1.024 ** 3 * 1.024 ** 3 / 1.0737))

    out = {"points": pts, "fit_seconds_a": a, "fit_seconds_per_row_us": b * 1e6,
           "fit_rss_a_mb": ma, "fit_rss_kb_per_row": mb,
           "row_bytes": 242.3}
    with open(os.path.join(WORK, "csv2h5_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("写入", os.path.join(WORK, "csv2h5_scaling.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

