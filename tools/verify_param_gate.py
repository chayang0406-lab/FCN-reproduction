"""端到端验证「参数量闸门」：真实建模型 -> 核对参数量 -> 故意传错也要报错。

这是给 runner 的示范：实验跑完就这么核一次。
既证明公式与真实模型一致，也证明闸门在配置不符时确实会拦下来
（否则「闸门」只是摆设）。
"""
import os
import sys

REMOTE = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118"
sys.path.insert(0, os.path.join(REMOTE, "tools"))
sys.path.insert(0, os.path.join(REMOTE, "code/FCN_repo"))

import torch
from fuxictr.features import FeatureMap
import src as model_zoo
from param_formula import count_params, check_model_params

BENCH = os.path.join(REMOTE, "bench")
D = os.path.join(BENCH, "data/Criteo_bench")


def build(exp, lin):
    params = dict(
        model_root=os.path.join(BENCH, "ckpt") + "/", verbose=0,
        early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id="Criteo_bench",
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=8192,
        embedding_dim=16, exp_num_layers=exp, lin_num_layers=lin,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=True, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
        train_data=os.path.join(D, "train.h5"),
        valid_data=os.path.join(D, "valid.h5"),
        test_data=os.path.join(D, "valid.h5"),
    )
    fmap = FeatureMap("Criteo_bench", D)
    fmap.load(os.path.join(D, "feature_map_bench.json"), params)
    return getattr(model_zoo, "FCNv2")(fmap, **params)


def main():
    fails = 0

    print("=" * 78)
    print("1. 真实模型 vs 闭式公式（多组配置）")
    print("=" * 78)
    for exp, lin in [(4, 1), (4, 2), (3, 3), (5, 5)]:
        m = build(exp, lin)
        expect = count_params(exp, lin)
        ok, actual, _, msg = check_model_params(m, expect, "exp=%d,lin=%d" % (exp, lin))
        print("  " + msg)
        if not ok:
            fails += 1
        del m
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("2. 闸门必须能拦住不符的配置（否则形同虚设）")
    print("=" * 78)
    m = build(4, 1)
    ok, actual, exp_, msg = check_model_params(m, 12345678, "故意传错")
    print("  " + msg)
    if ok:
        print("  [FAIL] 闸门没拦住错误期望值！")
        fails += 1
    else:
        print("  [OK] 闸门正确报出 MISMATCH")

    print()
    print("=" * 78)
    print("3. 模拟「跑错配置」：按 exp=4/lin=2 建模型，却拿 exp=4/lin=1 的期望值去核")
    print("=" * 78)
    del m
    torch.cuda.empty_cache()
    m2 = build(4, 2)
    ok, actual, exp_, msg = check_model_params(m2, count_params(4, 1), "跑错配置模拟")
    print("  " + msg)
    if ok:
        print("  [FAIL] 没发现配置不一致！")
        fails += 1
    else:
        print("  [OK] 成功发现「实际 16,924,434 而非期望 16,532,562」")

    print()
    print("=" * 78)
    print("RESULT:", "PASS —— 参数量闸门可用" if fails == 0 else "FAIL (%d 项)" % fails)
    print("=" * 78)
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

