"""测量 validation eval 的前向开销，用于估算「真实数据每 epoch 含 eval」的耗时。

背景：作者日志实测（tools/split_train_eval_time.py 拆解）
  _016 日志(exp=4/lin=1): train 190.5s/epoch, eval 13.3s/epoch, eval 占 6.5%
  主日志(旧版)          : train 178.9s/epoch, eval 11.7s/epoch, eval 占 6.1%

本脚本测的是本机 DCU 上 eval 前向的相对开销：
  分别测 train_step（前向+反向+step）与 eval 前向（仅前向、no_grad）的单 batch 时间，
  得到 eval/train 的比值，再套到真实规模上估算。

关键设计：必须用「作者真实 vocab 尺寸」建模型（16,532,562 参数），
否则 embedding 访存模式与真实差太远，测出来的比值没意义。
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
sys.path.insert(0, os.path.join(REMOTE, "tools"))

# 复用 captain 的造数逻辑：直接找已有的 bench 数据
def find_fm():
    for root, dirs, files in os.walk(WORK):
        for fn in files:
            if fn.startswith("feature_map") and fn.endswith(".json"):
                return root, os.path.join(root, fn)
    return None, None


def main():
    data_dir, fm_path = find_fm()
    assert data_dir, "找不到 bench 的 feature_map*.json，请先跑 tools/bench_dcu_throughput.py"
    print("使用 bench 数据目录:", data_dir)
    print("feature_map:", fm_path)

    params = dict(
        model_root=os.path.join(BENCH, "ckpt") + "/", verbose=1,
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

    n_params = sum(p.numel() for p in model.parameters())
    print("参数量 =", n_params)

    gen = H5DataLoader(fmap, stage="train", **params).make_iterator()[0]
    batch = next(iter(gen)).to(model.device)
    print("batch shape =", tuple(batch.shape))

    def time_train(n):
        for _ in range(20):
            model.train_step(batch)          # 预热
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model.train_step(batch)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    def time_eval(n):
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model.forward(batch)         # 预热
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                model.forward(batch)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n
        model.train()
        return dt

    tr = time_train(100)
    ev = time_eval(100)

    print()
    print("=" * 76)
    print("单 batch 耗时（batch_size=8192, 真实 vocab 规模）")
    print("=" * 76)
    print("  train_step (前向+反向+step) : %.4f s" % tr)
    print("  eval forward (仅前向 no_grad): %.4f s" % ev)
    print("  eval / train 比值           : %.3f" % (ev / tr))

    # 真实规模换算
    TRAIN_BATCHES = 4477          # 36,672,493 / 8192
    VALID_SAMPLES = 4584062
    for vb in (8192, 16384):
        nb = VALID_SAMPLES / vb
        est_train = tr * TRAIN_BATCHES
        est_eval = ev * nb
        print()
        print("  --- 假设 eval 用 batch_size=%d（%d 个 batch）---" % (vb, round(nb)))
        print("    train  = %.4f × 4477 = %.1f s" % (tr, est_train))
        print("    eval   = %.4f × %.0f   = %.1f s" % (ev, nb, est_eval))
        print("    合计/epoch            = %.1f s (%.2f min)" % (est_train + est_eval, (est_train + est_eval) / 60))
        print("    eval 占比             = %.1f%%" % (100 * est_eval / (est_train + est_eval)))

    print()
    print("=" * 76)
    print("对照作者 RTX 4090（日志实测拆解）")
    print("=" * 76)
    print("  _016 (exp=4/lin=1): train 190.5 s, eval 13.3 s, 合计 203.8 s, eval 占 6.5%")
    print("  主日志 (旧版)      : train 178.9 s, eval 11.7 s, 合计 190.6 s, eval 占 6.1%")

    out = {
        "n_params": n_params,
        "batch_size": 8192,
        "s_per_batch_train": round(tr, 6),
        "s_per_batch_eval": round(ev, 6),
        "eval_over_train_ratio": round(ev / tr, 4),
        "est_train_s": round(tr * TRAIN_BATCHES, 1),
        "author_train_s": 190.5,
        "author_eval_s": 13.3,
        "author_total_s": 203.8,
    }
    import json
    p = os.path.join(BENCH, "eval_overhead.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("写入", p)


if __name__ == "__main__":
    main()

