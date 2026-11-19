"""排查 train_step() 到底依赖哪些「只在 fit() 里才初始化」的属性。

capain 已定位 _max_gradient_norm。本脚本做两件事：
  1) 静态分析：把 train_step 调用链上读到的所有 self._xxx 都列出来，
     比对哪些在 __init__ 里赋值、哪些只在 fit() 里赋值。
  2) 实证验证：真建一个模型，分别试「不设 _max_gradient_norm」和「手动设上」，
     看能不能跑通，并把除了 _max_gradient_norm 之外还会炸的属性全试出来。
"""
import inspect
import os
import re
import sys

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
REPO = REMOTE + "/code/FCN_repo"
sys.path.insert(0, REPO)

FUXI = "/usr/local/lib/python3.11/site-packages/fuxictr"
BASE = os.path.join(FUXI, "pytorch/models/base_model.py")


# ---------- 第 1 部分：静态分析 ----------
def static_analysis():
    print("=" * 78)
    print("第 1 部分：静态分析 —— train_step 调用链读了哪些 self._xxx")
    print("=" * 78)

    src = open(BASE, encoding="utf-8").read()
    lines = src.split("\n")

    # 抓 __init__ 方法体
    def method_body(name):
        m = re.search(r"\n    def %s\(" % re.escape(name), src)
        if not m:
            return ""
        start = m.start()
        m2 = re.search(r"\n    def ", src[start + 10:])
        end = start + 10 + m2.start() if m2 else len(src)
        return src[start:end]

    init_body = method_body("__init__")
    fit_body = method_body("fit")
    ts_body = method_body("train_step")
    gel_body = method_body("get_total_loss")
    al_body = method_body("add_loss")
    ar_body = method_body("add_regularization")

    def assigned(body):
        return set(re.findall(r"self\.(_\w+)\s*=", body))

    def read(body):
        # 读不含赋值的出现
        return set(re.findall(r"self\.(_\w+)", body))

    init_set = assigned(init_body)
    fit_set = assigned(fit_body)

    print("  __init__ 里赋值的私有属性 (%d 个):" % len(init_set))
    print("   ", sorted(init_set))
    print()
    print("  fit() 里赋值的私有属性 (%d 个):" % len(fit_set))
    print("   ", sorted(fit_set))
    print()

    chain = {
        "train_step": read(ts_body),
        "get_total_loss": read(gel_body),
        "add_loss": read(al_body),
        "add_regularization": read(ar_body),
    }
    all_read = set()
    for k, v in chain.items():
        print("  %s 读到的 self._xxx: %s" % (k, sorted(v) if v else "(无)"))
        all_read |= v
    print()

    only_fit = sorted(all_read & fit_set - init_set)
    ok = sorted(all_read & init_set)
    print("  >>> 调用链需要、且【只在 fit() 里赋值】的：%s" % (only_fit or "(无)"))
    print("  >>> 调用链需要、且 __init__ 已赋值的      ：%s" % (ok or "(无)"))
    print()
    return only_fit


# ---------- 第 2 部分：实证 ----------
def empirical():
    print("=" * 78)
    print("第 2 部分：实证 —— 真建模型，试 train_step")
    print("=" * 78)
    import torch
    import numpy as np
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.dataloaders import H5DataLoader
    import src as model_zoo

    WORK = os.path.join(REMOTE, "scripts", "_smoke_remote")
    data_dir = os.path.join(WORK, "data", "Criteo_smoke")
    fm_json = os.path.join(data_dir, "feature_map.json")
    assert os.path.exists(fm_json), "先跑 smoke_remote_fcnv2.py 生成数据"

    params = dict(
        model_root=os.path.join(WORK, "checkpoints") + "/", verbose=1,
        early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id="Criteo_smoke",
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=512,
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

    gen = H5DataLoader(fmap, stage="train", **params).make_iterator()[0]
    batch = next(iter(gen))
    batch = batch.to(model.device)

    print()
    print("--- 测试 1: 不设 _max_gradient_norm，直接调 train_step ---")
    has = hasattr(model, "_max_gradient_norm")
    print("  hasattr(model, '_max_gradient_norm') =", has)
    try:
        model.train_step(batch)
        print("  结果: 竟然通过了（说明不需手动设）")
        test1 = "PASS"
    except Exception as e:
        print("  结果: 抛异常 %s: %s" % (type(e).__name__, e))
        test1 = "%s: %s" % (type(e).__name__, e)

    print()
    print("--- 测试 2: 手动设 _max_gradient_norm = 10.0 再调 train_step ---")
    model._max_gradient_norm = 10.0
    try:
        model.optimizer.zero_grad()
        loss = model.train_step(batch)
        torch.cuda.synchronize()
        print("  结果: 成功，loss = %.6f，requires_grad=%s" % (loss.item(), loss.requires_grad))
        test2 = "PASS"
    except Exception as e:
        print("  结果: 抛异常 %s: %s" % (type(e).__name__, e))
        test2 = "%s: %s" % (type(e).__name__, e)

    print()
    print("--- 测试 3: 连续多步，看还有没有别的坑 ---")
    errs = set()
    model._max_gradient_norm = 10.0
    for i in range(20):
        try:
            model.train_step(batch)
        except Exception as e:
            errs.add("%s: %s" % (type(e).__name__, e))
    torch.cuda.synchronize()
    print("  20 步连续 train_step，异常集合 = %s" % (errs or "空（全程无异常）"))

    print()
    print("--- 测试 4: 顺便看看 captain 提到的 _best_metric / _cur_epoch ---")
    for attr in ["_best_metric", "_cur_epoch", "_epoch_index", "_total_steps",
                 "_eval_steps", "_stop_training", "_batch_index", "_stopping_steps",
                 "_steps_per_epoch", "_max_gradient_norm"]:
        v = getattr(model, "_" + attr.lstrip("_"), "<不存在>")
        print("    self.%-20s = %r" % (attr, v))

    print()
    print("--- 测试 5: 稳态单 batch 耗时（预热 10 步后测 50 步）---")
    import time
    model._max_gradient_norm = 10.0
    for _ in range(10):
        model.train_step(batch)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 50
    for _ in range(N):
        model.train_step(batch)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print("  稳态 train_step 单次 = %.4f s  (batch=512, 参考值)" % dt)
    print("  按此速率估算作者规模(4477 batch/epoch) = %.1f s/epoch" % (dt * 4477))
    print("  注意: 合成数据 embedding 表小(vocab~50)，真实 Criteo vocab 91万，")
    print("        真实每 batch 会更慢；此数仅供参考，正式值以 bench 为准。")

    return test1, test2


if __name__ == "__main__":
    only_fit = static_analysis()
    t1, t2 = empirical()
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  静态分析：只在 fit() 里赋值、但 train_step 调用链需要的属性 =", only_fit or "(无)")
    print("  实证 测试1（不设）  =", t1)
    print("  实证 测试2（手动设）=", t2)

