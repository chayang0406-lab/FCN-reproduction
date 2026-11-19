"""用作者日志里的真实 feature_map 词表规模，验证 FCNv2 参数量能否精确复现 16,138,194。

这一步不依赖真实数据（不需要 4GB Criteo），但能证明：
  「环境 + fuxictr + FCNv2 源码 + 作者超参」这四者拼起来，得到的是和论文完全一致的模型，
  参数量逐位相同。如果这里对不上，说明环境或配置有问题，后面跑真实数据也白搭。

作者的 feature_map 取自 checkpoints/Criteo_x4_10_h5/FCNv2_Criteo.log。
"""
import json
import os
import sys
import tempfile

import torch

REPO = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo"
sys.path.insert(0, REPO)

# 作者日志里的词表规模（num_buckets）
AUTHOR_VOCAB = {
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18, "C7": 11882,
    "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656, "C13": 3176, "C14": 28,
    "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560, "C19": 2003, "C20": 5, "C21": 154564,
    "C22": 18, "C23": 17, "C24": 53031, "C25": 82, "C26": 40955,
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80, "I8": 69,
    "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
}

AUTHOR_N_PARAMS = 16138194
AUTHOR_HYPER = dict(
    embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
    num_heads=1, batch_norm=False, layer_norm=False, net_dropout=0.1,
)

# 作者的顺序：C1..C26 然后 I1..I13
FEATURE_ORDER = ["C%d" % i for i in range(1, 27)] + ["I%d" % i for i in range(1, 14)]


def build_feature_map(data_dir):
    fm = {
        "dataset_id": "Criteo_x4_author_caliber",
        "num_fields": len(FEATURE_ORDER),
        "total_features": sum(AUTHOR_VOCAB.values()),
        "labels": ["Label"],
        "features": [],
    }
    for name in FEATURE_ORDER:
        fm["features"].append({
            name: {
                "type": "categorical",
                "source": "item",
                "vocab_size": AUTHOR_VOCAB[name],
                "embedding_dim": AUTHOR_HYPER["embedding_dim"],
            }
        })
    path = os.path.join(data_dir, "feature_map.json")
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fm, f, indent=2)
    return path


def main():
    from fuxictr.features import FeatureMap

    work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_param_check")
    data_dir = os.path.join(work, "data")
    os.makedirs(work, exist_ok=True)
    fm_json = build_feature_map(data_dir)

    params = dict(
        model_root=os.path.join(work, "ckpt") + "/",
        verbose=1, early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id="Criteo_x4_author_caliber",
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=1e-5, net_regularizer=0, batch_size=8192,
        embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=True, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
    )

    feature_map = FeatureMap("Criteo_x4_author_caliber", data_dir)
    feature_map.load(fm_json, params)

    print("=== feature_map 校验 ===")
    print("  num_fields        =", feature_map.num_fields, "(作者 39)")
    print("  total_features    =", feature_map.total_features)
    print("  sum_emb_out_dim   =", feature_map.sum_emb_out_dim(),
          "(应为 39 x 16 = 624)")

    import src as model_zoo
    model = getattr(model_zoo, "FCNv2")(feature_map, **params)
    n = sum(p.numel() for p in model.parameters())

    emb = sum(p.numel() for nm, p in model.named_parameters() if "embedding_layer" in nm)
    net = n - emb

    print()
    print("=== 参数量对照 ===")
    print("  实测 总参数量     = %d" % n)
    print("  作者日志 参数量   = %d" % AUTHOR_N_PARAMS)
    print("  差值              = %d" % (n - AUTHOR_N_PARAMS))
    print("  其中 embedding 部分 = %d" % emb)
    print("  其中 cross-net 部分 = %d" % net)
    print()
    if n == AUTHOR_N_PARAMS:
        print("PASS: 参数量与作者日志逐位一致（%d）" % n)
        print("      => 环境 + fuxictr + FCNv2 源码 + 作者超参 = 论文同款模型")
        rc = 0
    else:
        print("FAIL: 参数量不一致，说明配置或源码有偏差")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

