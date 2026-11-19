"""修正版吞吐测量：不同 batch_size 下的稳态单步耗时 + 正确的 s/epoch 估算。

要点：作者日志「4477 batches/epoch」是 batch_size=8192 下的批次数。
      不能用 batch=512 的单步耗时去乘 4477 —— 那是单位不匹配。
      必须 (a) 在相同 batch_size 下测，或 (b) 用 samples/s 换算。
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

WORK = os.path.join(REMOTE, "scripts", "_smoke_remote")
data_dir = os.path.join(WORK, "data", "Criteo_smoke")
fm_json = os.path.join(data_dir, "feature_map.json")

TRAIN_SAMPLES = 36672493   # 作者 Criteo train 条数


def build(batch_size):
    params = dict(
        model_root=os.path.join(WORK, "checkpoints") + "/", verbose=1,
        early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id="Criteo_smoke",
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=batch_size,
        embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=True, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
        train_data=os.path.join(data_dir, "train.h5"),
        valid_data=os.path.join(data_dir, "valid.h5"),
        test_data=os.path.join(data_dir, "test.h5"),
    )
    fmap = FeatureMap("Criteo_smoke", data_dir)
    fmap.load(fm_json, params)
    model = getattr(model_zoo, "FCNv2")(fmap, **params)
    model._max_gradient_norm = 10.0     # <-- 唯一需要手动补的一行
    gen = H5DataLoader(fmap, stage="train", **params).make_iterator()[0]
    return model, gen


print("=" * 84)
print("不同 batch_size 下的稳态 train_step 耗时")
print("=" * 84)
print("%-12s %-12s %-14s %-14s %s" % ("batch_size", "单步(s)", "samples/s", "估算 s/epoch", "备注"))

results = {}
for bs in (512, 2048, 8192):
    model, gen = build(bs)
    batch = next(iter(gen)).to(model.device)
    for _ in range(10):                       # 预热（含首次 kernel 编译）
        model.train_step(batch)
    torch.cuda.synchronize()
    N = 30
    t0 = time.time()
    for _ in range(N):
        model.train_step(batch)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    sps = bs / dt
    est = TRAIN_SAMPLES / sps
    results[bs] = (dt, sps, est)
    note = "<- 与作者同 batch_size" if bs == 8192 else ""
    print("%-12d %-12.4f %-14.0f %-14.1f %s" % (bs, dt, sps, est, note))

print()
print("=" * 84)
print("关键提醒：只有 batch_size=8192 那一行可以和作者 191 s/epoch 直接比")
print("=" * 84)
dt, sps, est = results[8192]
print("  batch=8192 估算 %.1f s/epoch（仅按 train_step，不含 validation）" % est)
print("  作者实测 191 s/epoch（含 train+eval，RTX 4090）")
print("  比值 = %.2fx" % (est / 191))
print()
print("  !! 重要限制（不可忽略）!!")
print("  1) 本测量用的是【合成数据】，embedding 表 vocab 只有约 50 个/特征，")
print("     而真实 Criteo vocab 合计 910,747（embedding 参数 14,571,952 个 ≈ 55.6 MB fp32）。")
print("     真实数据下 embedding 访存压力大得多，实际会更慢。")
print("  2) 本测量只含 train_step，不含每 epoch 的 validation evaluation。")
print("  3) 因此这个数只是【下界估计】，不能当 s/epoch 正式值用。")
print("     正式值必须在真实 Criteo h5 上测。")

