"""正确测量 eval 的规模扩展：分离固定开销与每样本边际成本。

【踩坑记录（重要，会再踩）】
前一版为了造大 valid 集，用 `src[c].shape[0]` 当 vocab 上界生成随机索引。
但 h5 里每个 dataset 的 shape[0] 是**行数**（50000），不是该列的 vocab_size。
于是给 vocab=1446 的列生成了 [0,50000) 的索引 -> DCU embedding 查表越界
-> 触发 HW KERNEL VMFault（不是 Python 异常，是硬件级报错，进程直接崩）。
正确做法：vocab 上界必须从 feature_map.json 的 vocab_size 字段读。

本脚本：
  1. 从 feature_map 读真实 vocab_size，安全生成多规模 valid h5
  2. 每个规模测 evaluate（含 h5 读 + 前向 + sklearn 指标）
  3. 线性拟合 out = a + b*n，a=固定开销，b=每样本边际成本
  4. 外推真实 458 万样本
"""
import json
import os
import sys
import time

import h5py
import numpy as np

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
sys.path.insert(0, REMOTE + "/code/FCN_repo")

import torch
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
import src as model_zoo

BENCH = os.path.join(REMOTE, "bench")
WORK = os.path.join(BENCH, "data")
REAL_VALID = 4584062
BATCH = 8192
TRAIN_BATCHES = 4477
S_PER_BATCH_TRAIN = 0.0426     # 本任务实测（bench_bs_sweep / captain bench 一致区间）


def find_fm():
    for root, dirs, files in os.walk(WORK):
        for fn in files:
            if fn.startswith("feature_map") and fn.endswith(".json"):
                return root, os.path.join(root, fn)
    return None, None


def make_valid_h5(out_path, n_rows, specs, seed=0):
    """specs: {col: vocab_size}。vocab 从 feature_map 读，保证不越界。"""
    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, "r") as f:
                if f.attrs.get("n_rows") == n_rows:
                    return out_path
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    with h5py.File(out_path, "w") as dst:
        for c, vocab in specs.items():
            if c == "Label":
                arr = rng.integers(0, 2, size=n_rows).astype("float32")
            else:
                # 留 1 个余量给 oov_idx
                hi = max(2, int(vocab) - 1)
                arr = rng.integers(0, hi, size=n_rows).astype("int32")
            dst.create_dataset(c, data=arr)
        dst.attrs["n_rows"] = n_rows
    return out_path


def main():
    data_dir, fm_path = find_fm()
    assert data_dir and fm_path
    params = dict(
        model_root=os.path.join(BENCH, "ckpt") + "/", verbose=0,
        early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id=os.path.basename(data_dir),
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=BATCH,
        embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=False, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
        train_data=os.path.join(data_dir, "train.h5"),
        valid_data=os.path.join(data_dir, "valid.h5"),
        test_data=os.path.join(data_dir, "valid.h5"),
    )
    fmap = FeatureMap(params["dataset_id"], data_dir)
    fmap.load(fm_path, params)
    model = getattr(model_zoo, "FCNv2")(fmap, **params)
    model._max_gradient_norm = 10.0
    n_params = sum(p.numel() for p in model.parameters())
    print("参数量 =", n_params)

    # 从 feature_map 读真实 vocab_size（关键修正）
    specs = {}
    for name, spec in fmap.features.items():
        specs[name] = int(spec.get("vocab_size", 2))
    specs["Label"] = 2
    print("字段数 =", len(fmap.features), " vocab 上界样例:",
          {k: specs[k] for k in list(fmap.features)[:3]})

    print()
    print("=" * 82)
    print("不同 valid 规模下的 evaluate 耗时")
    print("=" * 82)

    # 【必做】冷启动预热：第一次 evaluate 会带上 cuda context 建立、
    # sklearn 首次 import、h5 首次打开等一次性开销。
    # 不预热的话第一个测量点会严重偏高（实测 5 万行 4.23s > 20 万行 1.91s，明显倒挂）。
    warm_vp = make_valid_h5(os.path.join(data_dir, "v_warm.h5"), 50_000, specs, seed=999)
    _wp = dict(params)
    _wp["valid_data"] = warm_vp
    _wg = H5DataLoader(fmap, stage="train", **_wp).make_iterator()[1]
    model.evaluate(_wg)
    torch.cuda.synchronize()
    del _wg
    print("  (已用 5 万行预热一次，剔除冷启动开销)")
    print()
    print("%-12s %-9s %-12s %-13s %s" % ("n_rows", "batches", "evaluate(s)", "us/sample", "指标"))

    pts = []
    for n in (50_000, 200_000, 800_000):
        vp = make_valid_h5(os.path.join(data_dir, "v_%d.h5" % n), n, specs)
        p2 = dict(params)
        p2["valid_data"] = vp
        gen = H5DataLoader(fmap, stage="train", **p2).make_iterator()[1]
        torch.cuda.synchronize()
        t0 = time.time()
        res = model.evaluate(gen)
        torch.cuda.synchronize()
        dt = time.time() - t0
        pts.append((n, dt))
        print("%-12d %-9d %-12.2f %-13.2f ll=%.5f auc=%.5f"
              % (n, (n + BATCH - 1) // BATCH, dt, dt / n * 1e6, res["logloss"], res["AUC"]))
        del gen

    # 用最小与最大两个点做两点法（中间点受 batch 边界效应影响最小）
    (n1, t1), (n2, t2) = pts[0], pts[-1]
    b = (t2 - t1) / (n2 - n1)
    a = t1 - b * n1
    print()
    print("  两点法拟合(最小点与最大点): evaluate(s) = %.3f + %.6f * n_rows" % (a, b))
    print("    固定开销 a = %.2f s" % a)
    print("    边际成本 b = %.3f us/sample" % (b * 1e6))

    # 同时给最小二乘结果做对照，两者应接近
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    A = np.vstack([np.ones_like(xs), xs]).T
    (a_ls, b_ls), *_ = np.linalg.lstsq(A, ys, rcond=None)
    print("    最小二乘对照: a=%.3f, b=%.3f us/sample (差 %.1f%%)"
          % (a_ls, b_ls * 1e6, 100 * abs(b_ls - b) / b if b else 0))
    if b > 0:
        est = a + b * REAL_VALID
    else:
        est = a_ls + b_ls * REAL_VALID
    print("  外推真实 valid(%d) = %.1f s" % (REAL_VALID, est))

    train_s = S_PER_BATCH_TRAIN * TRAIN_BATCHES
    total = train_s + est

    print()
    print("=" * 82)
    print("DCU 每 epoch（含 eval）最终预估")
    print("=" * 82)
    print("  train = %.4f s/batch × %d = %.1f s" % (S_PER_BATCH_TRAIN, TRAIN_BATCHES, train_s))
    print("  eval  = %.1f s" % est)
    print("  合计  = %.1f s/epoch = %.2f min  (eval 占 %.1f%%)"
          % (total, total / 60, 100 * est / total))
    print()
    print("  作者 4090: train 190.5 + eval 13.3 = 203.8 s (eval 6.5%)")
    print("  DCU/4090 = %.2fx" % (total / 203.8))
    print()
    print("  20 epoch 单种子 = %.1f min" % (20 * total / 60))
    print("  7 种子          = %.1f h" % (7 * 20 * total / 3600))

    out = {
        "n_params": n_params,
        "batch_size": BATCH,
        "scaling_points": [{"n_rows": n, "evaluate_s": round(t, 2)} for n, t in pts],
        "fit_fixed_s": round(a, 3),
        "fit_per_sample_us": round(b * 1e6, 4),
        "est_eval_s": round(est, 1),
        "est_train_s": round(train_s, 1),
        "est_epoch_with_eval_s": round(total, 1),
        "eval_pct": round(100 * est / total, 1),
        "author_total_s": 203.8,
        "slowdown_vs_4090": round(total / 203.8, 2),
        "single_seed_20ep_min": round(20 * total / 60, 1),
        "seven_seeds_h": round(7 * 20 * total / 3600, 2),
    }
    p = os.path.join(BENCH, "eval_scaling.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("写入", p)


if __name__ == "__main__":
    main()

