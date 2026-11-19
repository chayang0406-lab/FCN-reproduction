#!/usr/bin/env python3
"""测量 DCU 上 FCNv2 的稳态训练吞吐，用于排期决策。

为什么必须先测：作者在 RTX 4090 上是 182 秒/epoch。如果 DCU 上慢 10 倍，
「7 个随机种子」的计划直接不成立，必须当场改设计。这个数字决定后面所有排期。

做法：
1. 生成与真实 Criteo_x4_10_h5 **同形状**的合成 h5（39 字段、910,747 总基数是假的，
   但这里的 embedding 表大小对耗时影响很小，主要耗时是 4477 个 batch 的算子本身）。
2. 用作者超参（batch 8192, exp=4, lin=1, emb_dim 16）跑若干 batch。
3. 丢掉前 N 个 batch（kernel 编译 + 预热），只统计稳态。
4. 输出：稳态 s/batch → 推算 s/epoch（4477 batch），并给出一个 epoch 的预估。

用法：
  bash run_bench.sh  （见同目录脚本）
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import h5py


def build_synthetic_h5(out_dir, n_train, n_valid, seed=0):
    """造与 Criteo_x4_10_h5 同字段结构的 h5，供计时用。

    关键：vocab_size 必须用**作者日志里的真实值**，否则 embedding 表大小失真，
    参数个数和 Adam 的稠密更新开销都会跟着变，测出来的耗时就不可信。
    """
    os.makedirs(out_dir, exist_ok=True)
    names = ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(1, 27)]
    # 逐字抄自作者日志 FCNv2_Criteo.log 的 Feature specs
    REAL_VOCAB = {
        "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
        "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
        "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
        "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
        "C25": 82, "C26": 40955,
        "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
        "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
    }
    rng = np.random.default_rng(seed)
    vocab = {n: REAL_VOCAB[n] for n in names}

    meta = {
        "dataset_id": "Criteo_bench",
        "labels": ["Label"],
        "num_fields": len(names),
        # 注意：fuxictr 2.0.1 的 FeatureMap.load 期望 features 是
        # 「每项只有一个键的 dict」组成的 list，不是一个大 dict。
        "features": [
            {n: {
                "source": "",
                "type": "categorical",
                "padding_idx": 0,
                "oov_idx": vocab[n] - 1,
                "vocab_size": vocab[n],
            }}
            for n in names
        ],
    }
    with open(os.path.join(out_dir, "feature_map_bench.json"), "w") as f:
        json.dump(meta, f)

    for split, n in (("train", n_train), ("valid", n_valid)):
        p = os.path.join(out_dir, "%s.h5" % split)
        if os.path.exists(p):
            os.remove(p)
        with h5py.File(p, "w") as h5:
            h5.create_dataset("Label", data=rng.integers(0, 2, size=n).astype(np.float32))
            for nm in names:
                v = vocab[nm]
                h5.create_dataset(
                    nm, data=rng.integers(1, v, size=n, dtype=np.int64)
                )
    return os.path.join(out_dir, "feature_map_bench.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FCN 仓库路径")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n-train", type=int, default=400000)
    ap.add_argument("--n-valid", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=40, help="预热 batch 数（丢弃）")
    ap.add_argument("--measure", type=int, default=120, help="计时 batch 数")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)

    import torch
    from fuxictr.utils import load_config, set_logger
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.dataloaders import H5DataLoader
    from fuxictr.pytorch.torch_utils import seed_everything

    import src as model_zoo

    work = args.workdir
    os.makedirs(work, exist_ok=True)
    data_dir = os.path.join(work, "data", "Criteo_bench")
    print("[1/4] 造合成 h5（train=%d valid=%d）..." % (args.n_train, args.n_valid), flush=True)
    # 支撑 warmup + measure 所需的样本数，并留 1.5 倍余量
    need = int((args.warmup + args.measure) * args.batch_size * 1.5)
    if args.n_train < need:
        print("      n_train=%d 不足以支撑 %d batch，自动提到 %d" %
              (args.n_train, args.warmup + args.measure, need), flush=True)
        args.n_train = need
    t0 = time.time()
    build_synthetic_h5(data_dir, args.n_train, args.n_valid)
    print("      造数耗时 %.1fs" % (time.time() - t0), flush=True)

    # 用作者的 FCNv2_Criteo 超参，只把数据换成合成数据
    params = {
        "model_root": os.path.join(work, "checkpoints") + "/",
        "num_workers": 4,
        "verbose": 0,
        "early_stop_patience": 2,
        "pickle_feature_encoder": True,
        "save_best_only": True,
        "eval_steps": None,
        "debug_mode": False,
        "group_id": None,
        "use_features": None,
        "feature_specs": None,
        "feature_config": None,
        "model": "FCNv2",
        "dataset_id": "Criteo_bench",
        "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"],
        "task": "binary_classification",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "embedding_regularizer": 1e-5,
        "net_regularizer": 0,
        "batch_size": args.batch_size,
        "embedding_dim": 16,
        "exp_num_layers": 4,
        "lin_num_layers": 1,
        "net_dropout": 0.1,
        "batch_norm": False,
        "layer_norm": False,
        "num_heads": 1,
        "epochs": 1,
        "shuffle": True,
        "seed": 2025,
        "monitor": {"AUC": 1, "logloss": 0},
        "monitor_mode": "max",
        "gpu": 0,
        "data_format": "h5",
        "data_root": os.path.join(work, "data") + "/",
        "train_data": os.path.join(data_dir, "train.h5"),
        "valid_data": os.path.join(data_dir, "valid.h5"),
        "test_data": os.path.join(data_dir, "valid.h5"),
        "label_col": {"dtype": "float", "name": "Label"},
        "feature_cols": [{"active": True, "dtype": "str", "name": n, "type": "categorical"}
                         for n in (["I%d" % i for i in range(1, 14)] +
                                   ["C%d" % i for i in range(1, 27)])],
    }
    # feature_map 直接从合成时写的 json 载入
    fm_path = os.path.join(data_dir, "feature_map_bench.json")
    feature_map = FeatureMap("Criteo_bench", data_dir)
    feature_map.load(fm_path, params)

    print("[2/4] 建模型...", flush=True)
    seed_everything(seed=2025)
    model_class = getattr(model_zoo, "FCNv2")
    model = model_class(feature_map, **params)
    n_params = sum(p.numel() for p in model.parameters())
    print("      参数量 = %d ; device = %s" % (n_params, model.device), flush=True)

    print("[3/4] 造 dataloader...", flush=True)
    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()

    # fuxictr 2.0.1 的设计缺陷：train_step() 依赖 self._max_gradient_norm，
    # 但这个属性只在 fit() 里被赋值（base_model.py 第 139 行），__init__ 里没有。
    # 所以不先调 fit() 就直接调 train_step() 会 AttributeError。
    # 这里手工补上 fit() 会设置的那几个属性，从而走**与真实训练完全相同**的
    # train_epoch → train_step 代码路径（而不是自己重写前向反向）。
    model._max_gradient_norm = 10.0
    model._total_steps = 0
    model._batch_index = 0
    model._eval_steps = 10 ** 12   # 让它不要去 eval，只测纯训练吞吐
    model.train()

    print("[4/4] 计时：预热 %d batch，然后测 %d batch" % (args.warmup, args.measure), flush=True)
    it = iter(train_gen)
    for i in range(args.warmup):
        batch = next(it)
        model.train_step(batch)
        if (i + 1) % 10 == 0:
            print("      warmup %d/%d" % (i + 1, args.warmup), flush=True)
    torch.cuda.synchronize()

    t_start = time.time()
    for i in range(args.measure):
        batch = next(it)
        model.train_step(batch)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    s_per_batch = elapsed / args.measure
    # 真实 Criteo_x4_10_h5: 36,672,493 样本 / batch 8192 = 4477 batch
    real_batches = 4477
    s_per_epoch = s_per_batch * real_batches

    print("\n================ 结果 ================")
    print("稳态 s/batch          : %.4f" % s_per_batch)
    print("推算 s/epoch (4477批) : %.1f  (%.2f 分钟)" % (s_per_epoch, s_per_epoch / 60))
    print("作者 RTX 4090 参考     : 182 s")
    print("倍数                  : %.2fx" % (s_per_epoch / 182))
    print("单种子 20 epoch 预估   : %.1f 分钟" % (s_per_epoch * 20 / 60))
    print("7 种子 全流程预估      : %.1f 小时" % (s_per_epoch * 20 * 7 / 3600))
    print("======================================")

    out = {
        "n_params": n_params,
        "batch_size": args.batch_size,
        "s_per_batch_steady": s_per_batch,
        "real_batches_per_epoch": real_batches,
        "s_per_epoch_est": s_per_epoch,
        "author_4090_s_per_epoch": 182,
        "slowdown_x": s_per_epoch / 182,
        "single_seed_20ep_min": s_per_epoch * 20 / 60,
        "seven_seeds_hours": s_per_epoch * 20 * 7 / 3600,
        "warmup_batches": args.warmup,
        "measured_batches": args.measure,
    }
    with open(os.path.join(work, "bench_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n写入 %s" % os.path.join(work, "bench_result.json"))


if __name__ == "__main__":
    main()

