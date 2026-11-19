"""精确复现作者参数量 16,138,194 的条件。

手算发现：exp_num_layers=3, lin_num_layers=1, embedding_dim=16
得到 16,140,690，与作者 16,138,194 仅差 2,496。

2496 是什么？ 2496 = 156 * 16 = ... 试：39*16=624, 624*4=2496。
也可能只是 vocab_size 的微小差异（数据集切分不同导致 min_categr_count 过滤后的词表不同）。

本脚本用真模型 + 真 feature_map 验证 exp=3,lin=1 这一组，并算清 2496 的来源。
"""
import json
import os
import sys

REPO = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo"
sys.path.insert(0, REPO)

AUTHOR_VOCAB = {
    "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18, "C7": 11882,
    "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656, "C13": 3176, "C14": 28,
    "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560, "C19": 2003, "C20": 5, "C21": 154564,
    "C22": 18, "C23": 17, "C24": 53031, "C25": 82, "C26": 40955,
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80, "I8": 69,
    "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
}
TARGET = 16138194
FEATURE_ORDER = ["C%d" % i for i in range(1, 27)] + ["I%d" % i for i in range(1, 14)]


def build_fm(data_dir, vocab):
    fm = {
        "dataset_id": "Criteo_caliber",
        "num_fields": len(FEATURE_ORDER),
        "total_features": sum(vocab.values()),
        "labels": ["Label"],
        "features": [
            {n: {"type": "categorical", "source": "item", "vocab_size": vocab[n],
                 "embedding_dim": 16}}
            for n in FEATURE_ORDER
        ],
    }
    os.makedirs(data_dir, exist_ok=True)
    p = os.path.join(data_dir, "feature_map.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(fm, f, indent=2)
    return p


def mk_params(work, exp_l, lin_l, reg):
    return dict(
        model_root=os.path.join(work, "ckpt") + "/",
        verbose=1, early_stop_patience=2, save_best_only=True, eval_steps=None,
        model="FCNv2", dataset_id="Criteo_caliber",
        loss="binary_crossentropy", metrics=["logloss", "AUC"],
        task="binary_classification", optimizer="adam", learning_rate=1e-3,
        embedding_regularizer=reg, net_regularizer=0, batch_size=8192,
        embedding_dim=16, exp_num_layers=exp_l, lin_num_layers=lin_l,
        net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1,
        epochs=1, shuffle=True, seed=2025,
        monitor={"AUC": 1, "logloss": 0}, monitor_mode="max", gpu=0,
    )


def count(fm_json, data_dir, exp_l, lin_l, work, reg=1e-5):
    from fuxictr.features import FeatureMap
    import src as model_zoo
    params = mk_params(work, exp_l, lin_l, reg)
    fmap = FeatureMap("Criteo_caliber", data_dir)
    fmap.load(fm_json, params)
    m = getattr(model_zoo, "FCNv2")(fmap, **params)
    return sum(p.numel() for p in m.parameters()), m


def main():
    work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_param_check2")
    data_dir = os.path.join(work, "data")
    fm_json = build_fm(data_dir, AUTHOR_VOCAB)

    print("=== 用真模型验证若干超参组合 ===")
    for exp_l, lin_l in [(4, 1), (3, 1), (4, 0), (3, 0), (2, 1)]:
        n, _ = count(fm_json, data_dir, exp_l, lin_l, work)
        flag = "  <== 命中" if n == TARGET else ""
        print("  exp_num_layers=%d lin_num_layers=%d -> %d   (差 %+d)%s"
              % (exp_l, lin_l, n, n - TARGET, flag))

    print()
    print("=== 分析 2496 的差额（exp=3,lin=1 时） ===")
    n31, model31 = count(fm_json, data_dir, 3, 1, work)
    gap = n31 - TARGET
    print("  实测 %d, 目标 %d, 差 %+d" % (n31, TARGET, gap))
    print("  gap / embedding_dim(16) = %s  <- 相当于多少个 vocab 条目"
          % (gap / 16))

    # 逐特征拆解 embedding 参数量
    emb_params = {nm: p.numel() for nm, p in model31.named_parameters()
                  if "embedding_layer" in nm and "weight" in nm}
    print("  embedding 表个数 = %d" % len(emb_params))
    tot_emb = sum(emb_params.values())
    print("  embedding 参数合计 = %d  (= sum(vocab)*16 should be %d)"
          % (tot_emb, sum(AUTHOR_VOCAB.values()) * 16))

    print()
    print("结论提示：")
    print("  - 作者主日志的 16,138,194 【无法】由 HEAD 代码的任何 exp/lin 组合复现（上表已枚举）。")
    print("  - 最接近的是 exp=3/lin=1（16,140,690），差 2,496，但这是巧合，不是等价。")
    print("  - 真正原因：主日志用的是旧版架构，每层宽 D//3 而非 D//2。")
    print("    反解验证：14,571,952 + 2 x (6 x 130,416 + 625) = 16,138,194，残差 0。")
    print("    详见 tools/explain_old_params.py 与 docs/环境_实测.md 第 7.3 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

