# -*- coding: utf-8 -*-
"""测试夹具：在没有真实数据、没有磁盘上 h5 的情况下造出可用的 FeatureMap 与假批次。

为什么不用真实 Criteo h5：
    这些测试要能在**任何机器**上 30 秒内跑完（本地 Windows、远程 DCU 都要能跑），
    目的是证明"开关关闭 ⇔ 与原仓库逐位一致"这个逻辑性质，
    与数据本身无关。真实数据的端到端一致性由 smoke 脚本另行负责。

但词表大小**照抄作者日志里的真实 vocab_size**（C1=1446、C3=157339 …），
这样 input_dim = 39 字段 × 16 = 624，与作者配置完全一致，
避免"用一个不真实的小模型测出通过、真跑却不一样"的空头验收。
"""

from __future__ import absolute_import

import io
import json
import os

import torch

# 作者日志 checkpoints/Criteo_x4_10_h5/FCNv2_Criteo.log 里 Feature specs 的原值。
# 抄真值而不是随手编：保证 input_dim 与参数量对账都能对上真实配置。
CRITEO_VOCAB = {
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
    "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
    "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
    "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
    "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
    "C25": 82, "C26": 40955,
}

# 字段顺序必须与作者一致：先 13 个 I，后 26 个 C（共 39 个）
FEATURE_ORDER = ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(1, 27)]


def write_feature_map_json(data_dir, dataset_id="Criteo_x4_10_h5", order=None,
                           vocab=None):
    """在 data_dir 写一份 feature_map.json（格式与 fuxictr FeatureMap.save 一致）。"""
    order = list(order or FEATURE_ORDER)
    vocab = dict(vocab or CRITEO_VOCAB)
    os.makedirs(data_dir, exist_ok=True)
    features = []
    for name in order:
        vs = int(vocab[name])
        features.append({name: {
            "source": "",
            "type": "categorical",
            "padding_idx": 0,
            "oov_idx": vs - 1,
            "vocab_size": vs,
        }})
    payload = {
        "dataset_id": dataset_id,
        "num_fields": len(order),
        "total_features": sum(int(vocab[n]) for n in order),
        "input_length": len(order),
        "labels": ["Label"],
        "features": features,
    }
    path = os.path.join(data_dir, "feature_map.json")
    with io.open(path, "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=4)
    return path


def load_feature_map(data_dir, dataset_id="Criteo_x4_10_h5", embedding_dim=16):
    """构造并 load 一个 FeatureMap 对象。"""
    from fuxictr.features import FeatureMap

    params = {"dataset_id": dataset_id, "embedding_dim": embedding_dim,
              "ordered_features": None, "feature_specs": None}
    fm = FeatureMap(dataset_id, data_dir)
    fm.load(os.path.join(data_dir, "feature_map.json"), params)
    return fm


def make_batch(feature_map, batch_size=8, seed=0, spread=True):
    """造一个形状为 (B, input_length) 的整数批次，最后一列是 0/1 标签。

    spread=True 时故意让取值**铺满整个词表**而不只取 0/1：
        如果只取 0/1，所有样本会命中同一批 embedding 行，
        很多形状/广播错误会被掩盖（比如 b 的宽度错了一维也照样能算）。
    """
    g = torch.Generator().manual_seed(seed)
    cols = []
    for feature, spec in feature_map.features.items():
        vs = int(spec["vocab_size"])
        if spread:
            col = torch.randint(0, vs, (batch_size,), generator=g)
        else:
            col = torch.randint(0, 1, (batch_size,), generator=g)
        cols.append(col.float())
    # 标签：让正负各半，避免 loss 恰好对称导致某些项恒为 0 而测不出差别
    label = torch.tensor([[1.0] if i % 2 == 0 else [0.0] for i in range(batch_size)])
    batch = torch.stack(cols, dim=1)
    return torch.cat([batch, label], dim=1)


# 构造模型时 BaseModel 需要的那批 kwargs（与 run_expid 传进来的真实 params 同构）
def base_kwargs(model_root, model_id="FCNv2_bit_exact"):
    return dict(
        task="binary_classification",
        gpu=-1,
        monitor="AUC",
        save_best_only=True,
        monitor_mode="max",
        early_stop_patience=2,
        eval_steps=None,
        verbose=0,
        model_root=model_root,
        metrics=["logloss", "AUC"],
        optimizer="adam",
        loss="binary_crossentropy",
    )
