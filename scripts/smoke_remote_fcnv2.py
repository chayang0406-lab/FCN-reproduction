"""FCNv2 在 海光 DCU K500SM_AI 上的冒烟测试（远程版）。

目标：证明 FCNv2 能在 DCU 上完成一次真实的前向 + 反向，并把每一步耗时、设备信息写进日志。
不依赖真实 Criteo 数据，不联网。

与本地版（tools/smoke_local_fcnv2.py）的差别：
  1. gpu=0，模型、输入 batch 全部在 DCU 上
  2. 显式断言「梯度存在且有限」，不是只看 loss 打出来
  3. 逐步计时
"""
import os
import sys
import json
import time
import logging

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo"
WORK = os.path.join(HERE, "_smoke_remote")

TIMING = {}


def tic(key):
    TIMING[key] = {"start": time.time()}


def toc(key):
    if key not in TIMING:
        TIMING[key] = {}
    TIMING[key]["end"] = time.time()
    TIMING[key]["sec"] = round(TIMING[key]["end"] - TIMING[key]["start"], 3)
    print("  [time] %-28s %8.3f s" % (key, TIMING[key]["sec"]))


def make_synthetic_csv(path, n_rows, seed=0):
    """造一份 Criteo 形状的 CSV：13 个 I 字段 + 26 个 C 字段 + Label。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    cols = ["Label"] + ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(1, 27)]
    lines = [",".join(cols)]
    for _ in range(n_rows):
        y = int(rng.integers(0, 2))
        row = [str(y)]
        for _i in range(1, 14):
            v = int(rng.zipf(1.5)) if rng.random() < 0.8 else int(rng.integers(1, 5000))
            row.append(str(min(v, 10 ** 6)))
        for _i in range(1, 27):
            row.append("%08x" % int(rng.integers(0, 50)))
        lines.append(",".join(row))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 1


def build_config(cfg_dir):
    import yaml

    ds_id = "Criteo_smoke"
    data_root = os.path.join(WORK, "data") + "/"
    dataset_cfg = {
        "dataset_id": ds_id,
        "data_format": "csv",
        "data_root": data_root,
        "feature_cols": [
            {
                "active": True,
                "dtype": "float",
                "fill_na": 0,
                "na_value": 0,
                "name": ["I%d" % i for i in range(1, 14)],
                "preprocess": "convert_to_bucket",
                "type": "categorical",
            },
            {
                "active": True,
                "dtype": "str",
                "fill_na": "",
                "na_value": "",
                "name": ["C%d" % i for i in range(1, 27)],
                "type": "categorical",
            },
        ],
        "label_col": {"dtype": "float", "name": "Label"},
        "min_categr_count": 2,
        "train_data": os.path.join(WORK, "data", ds_id, "train.csv"),
        "valid_data": os.path.join(WORK, "data", ds_id, "valid.csv"),
        "test_data": os.path.join(WORK, "data", ds_id, "test.csv"),
    }
    model_cfg = {
        "model_root": os.path.join(WORK, "checkpoints") + "/",
        "num_workers": 0,
        "verbose": 1,
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
        "dataset_id": ds_id,
        "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"],
        "task": "binary_classification",
        "optimizer": "adam",
        "learning_rate": 1.0e-3,
        "embedding_regularizer": 1.0e-5,
        "net_regularizer": 0,
        "batch_size": 512,
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
    }
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "dataset_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({ds_id: dataset_cfg}, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(cfg_dir, "model_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"FCNv2_smoke": model_cfg}, f, allow_unicode=True, sort_keys=False)
    return cfg_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(WORK, exist_ok=True)

    print("=" * 78)
    print("[1/7] 环境与设备")
    tic("1_env")
    import torch
    import numpy as np
    import fuxictr
    import sklearn
    import h5py

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    dev_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    print("  python   ", sys.version.split()[0])
    print("  torch    ", torch.__version__)
    print("  fuxictr  ", fuxictr.__version__)
    print("  numpy    ", np.__version__)
    print("  sklearn  ", sklearn.__version__)
    print("  h5py     ", h5py.__version__)
    print("  cuda.is_available =", torch.cuda.is_available(), " device_count =", dev_count)
    print("  device   ", dev_name)
    if props is not None:
        print("  arch/gcnArchName =", getattr(props, "gcnArchName", "n/a"))
        print("  total_mem = %.2f GiB" % (props.total_memory / 1024 ** 3))
        print("  multiprocessor_count =", props.multi_processor_count)
    if not torch.cuda.is_available():
        print("!! DCU 不可用，本测试失败")
        return 2
    toc("1_env")

    print("[2/7] 造合成数据（极小）")
    tic("2_synth")
    n_tr = make_synthetic_csv(os.path.join(WORK, "data", "Criteo_smoke", "train.csv"), 3000, seed=1)
    n_va = make_synthetic_csv(os.path.join(WORK, "data", "Criteo_smoke", "valid.csv"), 600, seed=2)
    n_te = make_synthetic_csv(os.path.join(WORK, "data", "Criteo_smoke", "test.csv"), 600, seed=3)
    print("  train=%d valid=%d test=%d" % (n_tr, n_va, n_te))
    toc("2_synth")

    print("[3/7] 写 config")
    tic("3_config")
    cfg_dir = build_config(os.path.join(WORK, "config"))
    toc("3_config")

    print("[4/7] CSV -> h5（最易在新环境炸的一步）")
    tic("4_build_h5")
    from fuxictr.utils import load_config, set_logger
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.pytorch.dataloaders import H5DataLoader
    from fuxictr.preprocess import build_dataset
    # 【必须用 criteo 版】run_expid.py 用 kkbox 版会覆盖成没有 convert_to_bucket 的类
    from fuxictr.datasets.criteo import FeatureProcessor

    params = load_config(cfg_dir + "/", "FCNv2_smoke")
    params["gpu"] = 0
    set_logger(params)
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    os.makedirs(data_dir, exist_ok=True)
    fm_json = os.path.join(data_dir, "feature_map.json")
    if os.path.exists(fm_json):
        os.remove(fm_json)

    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(fm_json, params)
    print("  num_fields =", feature_map.num_fields, " input_length =", feature_map.input_length)
    print("  sum_emb_out_dim =", feature_map.sum_emb_out_dim())
    for k in ("train_data", "valid_data", "test_data"):
        print("    %s -> %s (%d bytes)" % (k, os.path.basename(params[k]), os.path.getsize(params[k])))
    toc("4_build_h5")

    print("[5/7] 在 DCU 上建 FCNv2")
    tic("5_build_model")
    sys.path.insert(0, REPO)
    import src as model_zoo
    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters()
    n_params = sum(p.numel() for p in model.parameters())
    print("  model.device =", model.device)
    print("  参数量 = %d" % n_params)
    p0 = next(model.parameters())
    print("  首个参数 device =", p0.device)
    if p0.device.type != "cuda":
        print("!! 模型没在 DCU 上")
        return 3
    toc("5_build_model")

    print("[6/7] 显式一次前向 + 反向（硬性验收点）")
    tic("6_fwd_bwd")
    train_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()[0]
    batch = next(iter(train_gen))
    print("  batch shape =", tuple(batch.shape), " dtype =", batch.dtype)
    batch = batch.to(model.device)

    model.train()
    model.optimizer.zero_grad()
    loss = model.get_total_loss(batch)
    scalar = loss.item()
    loss.backward()

    # 逐参数梯度检查：必须存在、有限、且不全为 0
    n_total = 0
    n_with_grad = 0
    n_nonzero = 0
    n_nonfinite = 0
    gnorm_sq = 0.0
    bad = []
    for name, p in model.named_parameters():
        n_total += 1
        if p.grad is None:
            bad.append(name)
            continue
        n_with_grad += 1
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            n_nonfinite += 1
        if g.abs().sum().item() > 0:
            n_nonzero += 1
        gnorm_sq += float((g.float() ** 2).sum().item())
    print("  loss = %.6f" % scalar)
    print("  参数张量数 = %d，拿到梯度 = %d，梯度含非零 = %d，含非有限值 = %d"
          % (n_total, n_with_grad, n_nonzero, n_nonfinite))
    print("  全局梯度 L2 范数 = %.6f" % (gnorm_sq ** 0.5))
    if bad:
        print("  无梯度的参数（前 5 个）:", bad[:5])

    assert loss.requires_grad and torch.isfinite(loss), "loss 异常"
    assert n_with_grad == n_total, "有参数没拿到梯度：%d/%d" % (n_with_grad, n_total)
    assert n_nonfinite == 0, "梯度里有 NaN/Inf"
    assert n_nonzero > 0, "梯度全为 0，反向没生效"
    assert gnorm_sq > 0, "全局梯度范数为 0"

    model.optimizer.step()
    torch.cuda.synchronize()
    toc("6_fwd_bwd")

    print("[7/7] 跑 1 个 epoch（fit 全程在 DCU 上）")
    tic("7_fit_epoch")
    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)
    res = model.evaluate(valid_gen)
    torch.cuda.synchronize()
    print("  valid 结果:", res)
    toc("7_fit_epoch")

    out = {
        "ok": True,
        "device_name": dev_name,
        "device_mem_GiB": round(props.total_memory / 1024 ** 3, 2),
        "gcnArchName": str(getattr(props, "gcnArchName", "n/a")),
        "torch": torch.__version__,
        "fuxictr": fuxictr.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "h5py": h5py.__version__,
        "n_params": n_params,
        "sum_emb_out_dim": feature_map.sum_emb_out_dim(),
        "num_fields": feature_map.num_fields,
        "fwd_bwd": {
            "loss": scalar,
            "params_total": n_total,
            "params_with_grad": n_with_grad,
            "params_nonzero_grad": n_nonzero,
            "nonfinite_grads": n_nonfinite,
            "grad_norm_l2": round(gnorm_sq ** 0.5, 6),
        },
        "valid_result": res,
        "timing_sec": {k: v.get("sec") for k, v in TIMING.items()},
    }
    with open(os.path.join(WORK, "smoke_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 78)
    print("PASS: FCNv2 在 %s 上完成了一次前向+反向，并跑完 1 个 epoch" % dev_name)
    print("结果文件:", os.path.join(WORK, "smoke_result.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

