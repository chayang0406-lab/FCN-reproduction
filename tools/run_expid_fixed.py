#!/usr/bin/env python3
"""修正版 run_expid：绕开作者仓库 run_expid.py 的三个真实缺陷。

作者仓库 `run_expid.py` 在实际环境下**无法直接运行**，已实测确认三个缺陷：

1. `from fuxictr.utils import ..., delete_model_files`
   → fuxictr 2.0.1 的 utils **没有** `delete_model_files`
     （只有 load_config / set_logger / print_to_json / print_to_list /
       load_h5 / load_model_config / load_dataset_config）
     → ImportError，脚本连启动都不行。

2. `from fuxictr.preprocess import FeatureProcessor`
   紧接着 `from fuxictr.datasets.kkbox import FeatureProcessor`
   → 前者被后者覆盖；KKBox 版没有 `convert_to_bucket`，
     Criteo 的 I 字段配置要求它有 → AttributeError。
     必须用 `from fuxictr.datasets.criteo import FeatureProcessor`。

3. `sys.path.append("/home/lhh/code")` 是作者本地路径，无害但应删。

本脚本复制了原作者的全部逻辑，只修上述三点，并把结果写成 JSON 便于汇总。

用法：
  python3 run_expid_fixed.py --config <config_dir> --expid <id> --gpu 0
                              [--result-json <out.json>]
"""
import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# 注意：绝不能用 fuxictr.datasets.kkbox 的 FeatureProcessor（没有 convert_to_bucket）
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import build_dataset
from fuxictr.datasets.criteo import FeatureProcessor   # ← 关键修正 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expid", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--repo", default=None, help="FCN 仓库根，用于 import src")
    args = ap.parse_args()

    if args.repo:
        sys.path.insert(0, args.repo)
    import src as model_zoo

    params = load_config(args.config, args.expid)
    params["gpu"] = args.gpu
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    if params.get("data_format") == "csv":
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)

    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters()
    n_params = sum(p.numel() for p in model.parameters())

    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

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

    out = {
        "expid": args.expid,
        "dataset_id": params["dataset_id"],
        "model": params["model"],
        "seed": params["seed"],
        "exp_num_layers": params.get("exp_num_layers"),
        "lin_num_layers": params.get("lin_num_layers"),
        "n_params": n_params,
        "valid": {k: float(v) for k, v in valid_result.items()} if hasattr(valid_result, "items") else str(valid_result),
        "test": {k: float(v) for k, v in test_result.items()} if hasattr(test_result, "items") else str(test_result),
        "feature_map_vocab_sum": int(sum(s["vocab_size"] for s in feature_map.features.values())),
        "n_fields": int(feature_map.num_fields),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 原脚本的写法：追加一行到 <config目录名>.csv
    result_filename = Path(args.config).name.replace(".yaml", "") + ".csv"
    with open(result_filename, "a+") as fw:
        fw.write(' %s,[command] python %s,[exp_id] %s,[dataset_id] %s,[train] N.A.,[val] %s,[test] %s\n'
                 % (datetime.now().strftime('%Y%m%d-%H%M%S'), ' '.join(sys.argv),
                    args.expid, params['dataset_id'],
                    print_to_list(valid_result), print_to_list(test_result)))

    if args.result_json:
        os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(out, f, indent=2)
        print("RESULT_JSON " + args.result_json)

    print("VALID %s" % json.dumps(out["valid"]))
    print("TEST  %s" % json.dumps(out["test"]))
    print("PARAMS %d" % n_params)


if __name__ == "__main__":
    main()

