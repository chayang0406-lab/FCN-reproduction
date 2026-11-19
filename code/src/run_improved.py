#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""改进实验的运行入口（直接跑 h5，不走 CSV 预处理，与原仓库 run_expid.py 行为一致）。

为什么另写一个 runner 而不是改 run_expid.py：
    1) run_expid.py 里有 `os.chdir(os.path.dirname(...))` 和硬编码的
       `sys.path.append("/home/lhh/code")`，那是作者本机路径，远程 DCU 上会误导；
    2) 它第 28-29 行先用通用 FeatureProcessor 又被 kkbox 版覆盖（Criteo 会炸），
       这个坑已在 infra 的冒烟测试里证实；h5 路径不触发，但留着容易误用；
    3) 我们要保证**原仓库文件一个字节都不改**，基线纯净；
    4) 需要把改进开关、种子、输出目录显式化，跑出来的日志才能进报告。

用法（远程 DCU）：
    source /opt/dtk-26.04/env.sh
    /usr/local/bin/python3 code/src/run_improved.py \
        --config code/configs --expid FCNv2_C_base --gpu 0 \
        --out runs/e3/e3_base_s2025

参数说明见 --help。所有参数都能被命令行覆盖，便于多种子扫描。
"""

from __future__ import absolute_import

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

# —— 让 `import src` 落到本目录（code/src），而不是原仓库 ——
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="跑 FCNv2 改进实验（原版/软权重/压缩率扫描）")
    p.add_argument("--config", required=True, help="配置目录（含 model_config.yaml 与 dataset_config.yaml）")
    p.add_argument("--expid", required=True, help="实验 id（model_config.yaml 里的键名）")
    p.add_argument("--gpu", type=int, default=0, help="GPU 序号，-1 为 CPU")
    p.add_argument("--out", default=None, help="输出目录（存日志与 result.json）；默认 runs/<expid>_<时间戳>")
    p.add_argument("--seed", type=int, default=None, help="覆盖配置里的 seed（多种子扫描用）")
    p.add_argument("--epochs", type=int, default=None, help="覆盖 epochs")
    p.add_argument("--data_root", default=None,
                   help="覆盖 data_root（h5 所在根目录）。远程数据路径与本地不同时必须显式指定，"
                        "不要把某个机器的绝对路径写进 yaml")
    p.add_argument("--model_root", default=None, help="覆盖 model_root（checkpoint 落盘目录）")
    # —— 两个改进开关，命令行可覆盖，便于不写新 yaml 也能扫参数 ——
    p.add_argument("--tri_bce_mode", default=None,
                   choices=["hard", "softplus", "smoothmax", "ratio", "plain"])
    p.add_argument("--tri_bce_alpha", type=float, default=None)
    p.add_argument("--tri_bce_beta", type=float, default=None)
    p.add_argument("--tri_bce_anneal_lambda", type=float, default=None)
    p.add_argument("--tri_bce_anneal_floor", type=float, default=None)
    p.add_argument("--lca_compress_ratio", type=float, default=None)
    p.add_argument("--verbose", type=int, default=None, help="0 关进度条")
    return p.parse_args(argv)


def _default_out(expid):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("runs", "%s_%s" % (expid, ts))


def _epochs_done(model, params):
    """实际跑完的 epoch 数（早停会提前结束，所以不能用配置里的 epochs）。

    BaseModel.fit 里 `self._epoch_index` 是"当前轮的下标"（从 0 开始），
    所以实际完成轮数 = 下标 + 1。
    """
    idx = getattr(model, "_epoch_index", None)
    if idx is None:
        return int(params.get("epochs") or 1)
    return int(idx) + 1


def main(argv=None):
    args = _parse_args(argv)
    out_dir = os.path.abspath(args.out or _default_out(args.expid))
    os.makedirs(out_dir, exist_ok=True)

    # —— 日志同时写文件与 stdout，方便 nohup 后台跑也能留痕 ——
    log_path = os.path.join(out_dir, "run.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

    # —— 延迟到日志就绪后再 import，避免 import 期告警丢进 stderr ——
    import numpy as np
    import torch
    from fuxictr.utils import load_config, set_logger, print_to_json
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.pytorch.dataloaders import H5DataLoader
    import src as model_zoo

    # numpy 2.x 删了 np.Inf 别名，而 fuxictr 2.0.1 的 base_model 用了它。
    # 远程 DCU 固定 numpy 1.25.0（有 np.Inf），本机是 2.x，为可移植性补一行；
    # 这里只补 Python 层的别名，不改 fuxictr 源码。
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    params = load_config(args.config, args.expid)
    params["gpu"] = args.gpu
    params["model_id"] = args.expid

    # —— 命令行覆盖配置里的值 ——
    overrides = {
        "seed": args.seed, "epochs": args.epochs, "verbose": args.verbose,
        "data_root": args.data_root, "model_root": args.model_root,
        "tri_bce_mode": args.tri_bce_mode, "tri_bce_alpha": args.tri_bce_alpha,
        "tri_bce_beta": args.tri_bce_beta,
        "tri_bce_anneal_lambda": args.tri_bce_anneal_lambda,
        "tri_bce_anneal_floor": args.tri_bce_anneal_floor,
        "lca_compress_ratio": args.lca_compress_ratio,
    }
    applied = {}
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
            applied[k] = v

    # 数据路径优先用配置里的显式 train_data/valid_data/test_data；
    # 若给了 --data_root 而配置里仍是相对路径，则把相对部分拼到新的根上。
    if args.data_root:
        for key in ("train_data", "valid_data", "test_data"):
            rel = str(params.get(key, "")).split("Criteo_x4_10_h5", 1)
            if len(rel) == 2:
                params[key] = os.path.join(args.data_root, "Criteo_x4_10_h5" + rel[1])
        params["data_root"] = args.data_root

    set_logger(params)
    logging.info("=" * 70)
    logging.info("运行参数：%s", print_to_json(params))
    logging.info("命令行覆盖项：%s", applied)
    logging.info("输出目录：%s", out_dir)

    seed_everything(seed=params["seed"])
    logging.info("seed = %s", params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("input_length=%d, sum_emb_out_dim=%d, num_fields=%d",
                 feature_map.input_length, feature_map.sum_emb_out_dim(),
                 feature_map.num_fields)

    model_class = getattr(model_zoo, params["model"])
    logging.info("模型类 = %s.%s", model_class.__module__, model_class.__name__)
    t0 = time.time()
    model = model_class(feature_map, **params)
    build_sec = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("模型构造耗时 %.2fs，总参数量 %d", build_sec, n_params)
    # 交叉网络单独报（报告里必须区分"交叉网络参数"和"总参数"两个口径）
    if hasattr(model, "E2LCN") and hasattr(model, "L2ECN"):
        n_cross = sum(p.numel() for p in model.E2LCN.parameters()) + \
                  sum(p.numel() for p in model.L2ECN.parameters())
        logging.info("交叉网络（E2L+L2E）参数量 %d，占总参数 %.2f%%",
                     n_cross, 100.0 * n_cross / max(n_params, 1))
    else:
        n_cross = None

    logging.info("开始加载数据（h5）...")
    t0 = time.time()
    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()
    n_batches = len(train_gen)
    logging.info("数据加载耗时 %.2fs，%d batch/epoch", time.time() - t0, n_batches)

    logging.info("开始训练")
    t0 = time.time()
    model.fit(train_gen, validation_data=valid_gen, **params)
    fit_sec = time.time() - t0
    logging.info("训练总耗时 %.1fs", fit_sec)

    logging.info("****** Validation evaluation ******")
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    logging.info("******** Test evaluation ********")
    test_gen = H5DataLoader(feature_map, stage="test", **params).make_iterator()
    test_result = {}
    model.testing = True
    if test_gen:
        test_result = model.evaluate(test_gen)

    result = {
        "expid": args.expid,
        "model_class": "%s.%s" % (model_class.__module__, model_class.__name__),
        "seed": params["seed"],
        "n_params_total": n_params,
        "n_params_cross": n_cross,
        "reported_epochs": params.get("epochs"),
        "epochs_done": _epochs_done(model, params),
        "batch_size": params.get("batch_size"),
        "batches_per_epoch": n_batches,
        "build_sec": round(build_sec, 3),
        "fit_sec": round(fit_sec, 3),
        "sec_per_epoch": round(fit_sec / max(1, _epochs_done(model, params)), 3),
        "valid": dict(valid_result),
        "test": dict(test_result),
        "improvements": {
            "tri_bce_mode": params.get("tri_bce_mode", "hard"),
            "tri_bce_alpha": params.get("tri_bce_alpha", 1.0),
            "tri_bce_beta": params.get("tri_bce_beta", 1.0),
            "tri_bce_anneal_lambda": params.get("tri_bce_anneal_lambda", 0.0),
            "tri_bce_anneal_floor": params.get("tri_bce_anneal_floor", 0.0),
            "lca_compress_ratio": params.get("lca_compress_ratio", 0.5),
        },
        "overrides_from_cli": applied,
        "gpu": args.gpu,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logging.info("结果已写入 %s", os.path.join(out_dir, "result.json"))
    logging.info("valid=%s", valid_result)
    logging.info("test=%s", test_result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
