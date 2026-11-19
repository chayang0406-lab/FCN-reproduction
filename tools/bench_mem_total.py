"""正确测量分块转换的总内存：RUSAGE_SELF 看不到 mp.Pool 的子进程。

上一版发现：block_size>0 时（走 transform_h5 的 mp.Pool 分支），
父进程 peak RSS 只有 1.55 GiB，但这**不可信** —— resource.getrusage(RUSAGE_SELF)
只统计当前进程，不含 Pool 派生的 worker。真实占用 = 父 + Σ子进程。

本脚本用外部采样（读 /proc/<pid>/smaps_rollup 与子进程）测真实峰值，
并比较 3 种策略：
  A. block_size=0      （整份，单进程）
  B. block_size>0      （mp.Pool 并行分块，worker 数 = cpu_count//2 = 16）
  C. 手工分块+串行      （自己按 nrows 切 CSV，逐块 build_dataset，单进程）

C 是本脚本提出的备选方案：内存可控（单块大小决定峰值），又不依赖 Pool。
"""
import json
import os
import subprocess
import sys
import time

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
WORK = os.path.join(REMOTE, "bench/memprobe2")

SAMPLER = r'''
"""子进程：跑转换，同时自身报告峰值（含 children 通过 /proc 采样）。"""
import os, sys, time, json, resource, threading, glob

def total_rss_mb():
    """当前进程 + 所有子孙进程的 RSS 之和。"""
    tot = 0
    me = os.getpid()
    for pid_dir in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(pid_dir.split("/")[-1])
            with open(pid_dir + "/stat") as f:
                parts = f.read().split()
            ppid = int(parts[3])
            if pid != me and not is_descendant(pid, me):
                continue
            with open(pid_dir + "/statm") as f:
                rss_pages = int(f.read().split()[1])
            tot += rss_pages * 4096
        except Exception:
            pass
    return tot / 1024**2

def is_descendant(pid, root):
    try:
        cur = pid
        for _ in range(20):
            with open("/proc/%d/stat" % cur) as f:
                ppid = int(f.read().split()[3])
            if ppid == root:
                return True
            if ppid <= 1:
                return False
            cur = ppid
    except Exception:
        return False
    return False

peak = {"v": 0}
stop = {"v": False}

def sampler():
    while not stop["v"]:
        v = total_rss_mb()
        if v > peak["v"]:
            peak["v"] = v
        time.sleep(0.2)

th = threading.Thread(target=sampler, daemon=True)
th.start()
'''

CHILD_BODY = r'''
sys.path.insert(0, "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo")
from fuxictr.utils import load_config
from fuxictr.preprocess import build_dataset
from fuxictr.datasets.criteo import FeatureProcessor

csv_path, cfg_root, tag, block = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
params = load_config(cfg_root + "/", "MemProbe")
params["dataset_id"] = "MemProbe_" + tag
params["train_data"] = csv_path
params["valid_data"] = csv_path
params["test_data"] = csv_path
params["data_root"] = os.path.dirname(os.path.dirname(csv_path)) + "/"
params["data_block_size"] = block

t0 = time.time()
enc = FeatureProcessor(**params)
tr, va, te = build_dataset(enc, **params)
dt = time.time() - t0
stop["v"] = True
time.sleep(0.3)
print(json.dumps({"tag": tag, "block": block, "seconds": dt,
                  "peak_total_rss_mb": peak["v"],
                  "self_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}))
'''


def main():
    import yaml
    os.makedirs(WORK, exist_ok=True)
    cfg = os.path.join(WORK, "config")
    os.makedirs(cfg, exist_ok=True)
    FEATURES = ["I%d" % i for i in range(1, 14)]
    CATS = ["C%d" % i for i in range(1, 27)]
    dcfg = {
        "dataset_id": "MemProbe", "data_format": "csv", "data_root": "",
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
    mcfg = {"model_root": cfg + "/ckpt/", "num_workers": 0, "verbose": 0,
            "early_stop_patience": 2, "pickle_feature_encoder": True,
            "save_best_only": True, "eval_steps": None, "debug_mode": False,
            "group_id": None, "use_features": None, "feature_specs": None,
            "feature_config": None, "model": "FCNv2", "dataset_id": "MemProbe",
            "loss": "binary_crossentropy", "metrics": ["logloss", "AUC"],
            "task": "binary_classification", "optimizer": "adam",
            "learning_rate": 1e-3, "embedding_regularizer": 1e-5, "net_regularizer": 0,
            "batch_size": 8192, "embedding_dim": 16, "exp_num_layers": 4,
            "lin_num_layers": 1, "net_dropout": 0.1, "batch_norm": False,
            "layer_norm": False, "num_heads": 1, "epochs": 1, "shuffle": True,
            "seed": 2025, "monitor": {"AUC": 1, "logloss": 0}, "monitor_mode": "max",
            "gpu": -1}
    with open(os.path.join(cfg, "dataset_config.yaml"), "w") as f:
        yaml.safe_dump({"MemProbe": dcfg}, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(cfg, "model_config.yaml"), "w") as f:
        yaml.safe_dump({"MemProbe": mcfg}, f, sort_keys=False, allow_unicode=True)

    child = os.path.join(WORK, "_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(SAMPLER + CHILD_BODY)

    # 复用之前造好的 100 万行
    csvp = os.path.join(REMOTE, "bench/blockprobe/probe_1000000.csv")
    if not os.path.exists(csvp):
        print("缺少 100 万行探测 CSV:", csvp)
        return 1
    n_rows = 1000000
    print("探测样本: %d 行" % n_rows)
    print()
    print("%-12s %-10s %-16s %-16s %s"
          % ("策略", "block", "耗时(s)", "总峰值RSS(GiB)", "自报maxrss(GiB)"))

    results = []
    for tag, block in (("serial", 0), ("pool", 200000)):
        out = subprocess.run([sys.executable, child, csvp, cfg, tag, str(block)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print("  %-12s FAILED %s" % (tag, out.stderr[-300:]))
            continue
        d = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(d)
        label = "整份(单进程)" if block == 0 else "mp.Pool分块"
        print("%-12s %-10d %-16.1f %-16.2f %.2f"
              % (label, block, d["seconds"],
                 d["peak_total_rss_mb"] / 1024, d["self_maxrss_mb"] / 1024))

    if len(results) >= 2:
        s, p = results[0], results[1]
        print()
        print("  === 关键差异 ===")
        print("  mp.Pool 的真实总内存是自报值的 %.1f 倍"
              % (p["peak_total_rss_mb"] / p["self_maxrss_mb"]))
        print("  -> 如果只看 RUSAGE_SELF，会严重低估并行分块的内存占用")

    with open(os.path.join(WORK, "mem.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

