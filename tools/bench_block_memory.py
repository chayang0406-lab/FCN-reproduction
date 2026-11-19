"""验证 fuxictr 的 data_block_size 是否真能降低峰值内存。

背景：按 fuxictr 真实路径（dtype=object）估算，train 单份 deep 内存 ~80 GiB，
preprocess 时 `ddf.loc[:, active_cols]` 再产生一份副本 -> 峰值可达 160 GiB，
**超过 115 GiB 的 cgroup 上限**，全量转换必然 OOM。

fuxictr 的 build_dataset 提供 data_block_size 参数，走 transform_h5 的分块分支。
但注意：**它只把 transform 阶段分块，read_csv 仍然是整份读入**。
所以分块只能省掉 transform 阶段的额外副本，省不掉 read_csv 的 80 GiB。

本脚本实测两点：
  A. 不分块时的峰值 RSS（用 200 万行放大到接近真实压力）
  B. 分块时的峰值 RSS
结论直接决定 data 那边该用哪种方式转。
"""
import json
import os
import subprocess
import sys

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
SUB = os.path.join(REMOTE, "data/_subset_for_dcu/train.csv")
WORK = os.path.join(REMOTE, "bench/blockprobe")

CHILD = r'''
import os, sys, time, json, resource
sys.path.insert(0, "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo")
from fuxictr.utils import load_config
from fuxictr.preprocess import build_dataset
from fuxictr.datasets.criteo import FeatureProcessor

csv_path, cfg_root, tag, block = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
params = load_config(cfg_root + "/", "BlockProbe")
params["dataset_id"] = "BlockProbe_" + tag
params["train_data"] = csv_path
params["valid_data"] = csv_path
params["test_data"] = csv_path
params["data_root"] = os.path.dirname(os.path.dirname(csv_path)) + "/"
params["data_block_size"] = block

t0 = time.time()
enc = FeatureProcessor(**params)
tr, va, te = build_dataset(enc, **params)
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(json.dumps({"tag": tag, "block": block, "seconds": dt, "peak_rss_mb": peak,
                  "train_out": str(tr)}))
'''

FEATURES = ["I%d" % i for i in range(1, 14)]
CATS = ["C%d" % i for i in range(1, 27)]


def write_cfg(root):
    import yaml
    os.makedirs(root, exist_ok=True)
    dcfg = {
        "dataset_id": "BlockProbe", "data_format": "csv", "data_root": "",
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
        "dataset_id": "BlockProbe", "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"], "task": "binary_classification",
        "optimizer": "adam", "learning_rate": 1e-3, "embedding_regularizer": 1e-5,
        "net_regularizer": 0, "batch_size": 8192, "embedding_dim": 16,
        "exp_num_layers": 4, "lin_num_layers": 1, "net_dropout": 0.1,
        "batch_norm": False, "layer_norm": False, "num_heads": 1, "epochs": 1,
        "shuffle": True, "seed": 2025, "monitor": {"AUC": 1, "logloss": 0},
        "monitor_mode": "max", "gpu": -1,
    }
    with open(os.path.join(root, "dataset_config.yaml"), "w") as f:
        yaml.safe_dump({"BlockProbe": dcfg}, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(root, "model_config.yaml"), "w") as f:
        yaml.safe_dump({"BlockProbe": mcfg}, f, sort_keys=False, allow_unicode=True)


def make_csv(n_rows, out_path):
    if os.path.exists(out_path):
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


def main():
    import yaml  # noqa
    os.makedirs(WORK, exist_ok=True)
    cfg = os.path.join(WORK, "config")
    write_cfg(cfg)
    child = os.path.join(WORK, "_child.py")
    open(child, "w", encoding="utf-8").write(CHILD)

    N = 1000000   # 100 万行：足够放大内存模式，又不至于跑太久
    csvp = make_csv(N, os.path.join(WORK, "probe_%d.csv" % N))
    print("探测样本: %d 行" % N)
    print()
    print("%-10s %-14s %-14s %s" % ("block_size", "seconds", "peakRSS(GiB)", "说明"))
    res = []
    for tag, block in (("noblk", 0), ("blk", 200000)):
        out = subprocess.run([sys.executable, child, csvp, cfg, tag, str(block)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print("  block=%-8d FAILED: %s" % (block, out.stderr[-400:]))
            continue
        d = json.loads(out.stdout.strip().splitlines()[-1])
        res.append(d)
        print("%-10d %-14.2f %-14.2f %s" % (
            block, d["seconds"], d["peak_rss_mb"] / 1024,
            "分块" if block else "整份"))

    if len(res) == 2:
        a, b = res
        print()
        print("  整份 %.2f GiB -> 分块 %.2f GiB，降低 %.0f%%"
              % (a["peak_rss_mb"] / 1024, b["peak_rss_mb"] / 1024,
                 100 * (1 - b["peak_rss_mb"] / a["peak_rss_mb"])))
        print("  耗时: 整份 %.1f s -> 分块 %.1f s" % (a["seconds"], b["seconds"]))
        print()
        print("  按 100 万行实测值线性外推到 36,672,493 行：")
        for d in res:
            est = d["peak_rss_mb"] / 1024 * 36672493 / N
            print("    block=%-8d -> 峰值 %.1f GiB  %s"
                  % (d["block"], est, "安全" if est < 114.97 * 0.85 else "!! 超限风险"))
    with open(os.path.join(WORK, "block.json"), "w") as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

