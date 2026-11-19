"""用真实路径测 eval 开销：走 model.evaluate() + H5DataLoader，含 h5 I/O 与 sklearn 指标。

前一版只测了 model.forward() 的裸前向（7.2ms/batch），但那不含：
  - h5 读盘（每 epoch 要把 4,584,062 × 40 列的 float64 从 h5 读进内存）
  - numpy hstack / torch.from_numpy 转换
  - sklearn roc_auc_score + log_loss 的计算
作者日志里 eval = 13.3s/epoch，如果只有前向那远用不了 13 秒，说明 I/O 是大头。

本脚本用 model.evaluate() 走完整路径，分别测：
  A. 裸前向（no_grad forward），纯算力
  B. model.evaluate(valid_gen)，完整口径（含 h5 读 + 指标计算）
再按真实 valid 样本数线性外推。
"""
import os
import sys
import time

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
sys.path.insert(0, REMOTE + "/code/FCN_repo")

import torch
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
import src as model_zoo

BENCH = os.path.join(REMOTE, "bench")
WORK = os.path.join(BENCH, "data")
REAL_VALID = 4584062


def find_fm():
    for root, dirs, files in os.walk(WORK):
        for fn in files:
            if fn.startswith("feature_map") and fn.endswith(".json"):
                return root, os.path.join(root, fn)
    return None, None


def main():
    data_dir, fm_path = find_fm()
    assert data_dir
    params = dict(
        model_root=os.path.join(BENCH, "ckpt") + "/", verbose=0,
        early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id=os.path.basename(data_dir),
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=8192,
        embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=False, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
        train_data=os.path.join(data_dir, "train.h5"),
        valid_data=os.path.join(data_dir, "valid.h5"),
        test_data=os.path.join(data_dir, "test.h5"),
    )
    fmap = FeatureMap(params["dataset_id"], data_dir)
    fmap.load(fm_path, params)
    model = getattr(model_zoo, "FCNv2")(fmap, **params)
    model._max_gradient_norm = 10.0
    print("参数量 =", sum(p.numel() for p in model.parameters()))

    # bench 只造了 train/valid，用 stage="both" 拿到 valid loaders
    # （作者日志里 eval 是评 valid 集，口径一致）
    _tr, valid_gen = H5DataLoader(fmap, stage="train", **params).make_iterator()
    n_valid_bench = len(valid_gen.dataset)
    print("bench valid 样本数 =", n_valid_bench, " batch 数 =", len(valid_gen))

    # A. 裸前向
    batch = next(iter(valid_gen)).to(model.device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model.forward(batch)
        torch.cuda.synchronize()
        t0 = time.time()
        N = 60
        for _ in range(N):
            model.forward(batch)
        torch.cuda.synchronize()
        fwd_only = (time.time() - t0) / N

    # B. 完整 evaluate（含 h5 读 + 指标）
    # 注意 Evaluate 每次都要重新读 h5，所以我们重建 loader 保证公平
    t0 = time.time()
    res = model.evaluate(valid_gen)
    torch.cuda.synchronize()
    eval_full = time.time() - t0
    print("evaluate 返回:", dict(res))

    per_sample_fwd = fwd_only / 8192
    per_sample_full = eval_full / n_valid_bench
    est_fwd = per_sample_fwd * REAL_VALID
    est_full = per_sample_full * REAL_VALID

    print()
    print("=" * 78)
    print("A. 裸前向（仅 GPU 算力，不含 h5 I/O 与指标计算）")
    print("=" * 78)
    print("  单 batch(8192) = %.4f s -> 每样本 %.3f us" % (fwd_only, per_sample_fwd * 1e6))
    print("  外推真实 valid(%d 样本) = %.1f s" % (REAL_VALID, est_fwd))

    print()
    print("=" * 78)
    print("B. 完整 evaluate（h5 读盘 + 转换 + 前向 + sklearn 指标）")
    print("=" * 78)
    print("  bench %d 样本实测 = %.2f s -> 每样本 %.3f us" % (n_valid_bench, eval_full, per_sample_full * 1e6))
    print("  外推真实 valid(%d 样本) = %.1f s" % (REAL_VALID, est_full))
    print()
    print("  => I/O + 指标 占 eval 总时间的 %.1f%%"
          % (100 * (1 - est_fwd / est_full) if est_full > 0 else 0))

    print()
    print("=" * 78)
    print("C. 作者 RTX 4090 实测（日志拆解，作对照）")
    print("=" * 78)
    print("  _016: train 190.5 s + eval 13.3 s = 203.8 s/epoch  (eval 占 6.5%)")

    print()
    print("=" * 78)
    print("D. DCU 上「每 epoch 含 eval」预估")
    print("=" * 78)
    train_s = 0.0426 * 4477     # 本次实测的 train_step
    print("  train = 0.0426 s/batch × 4477 = %.1f s" % train_s)
    print("  eval  = %.1f s（完整口径外推）" % est_full)
    print("  ----")
    print("  合计  = %.1f s/epoch = %.2f min" % (train_s + est_full, (train_s + est_full) / 60))
    print("  eval 占比 = %.1f%%" % (100 * est_full / (train_s + est_full)))

    import json
    out = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "batch_size": 8192,
        "fwd_only_s_per_batch": round(fwd_only, 6),
        "eval_full_s_per_bench_valid": round(eval_full, 3),
        "bench_valid_samples": n_valid_bench,
        "est_eval_fwd_only_s": round(est_fwd, 2),
        "est_eval_full_s": round(est_full, 2),
        "est_train_s": round(train_s, 1),
        "est_epoch_with_eval_s": round(train_s + est_full, 1),
        "author_train_s": 190.5,
        "author_eval_s": 13.3,
        "author_total_s": 203.8,
    }
    p = os.path.join(BENCH, "eval_overhead_full.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("写入", p)


if __name__ == "__main__":
    main()

