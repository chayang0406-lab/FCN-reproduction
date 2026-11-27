"""带线性 warmup + 余弦退火的训练入口。

替代 fuxictr 默认的"反应式衰减"（哪轮不涨就砍 10 倍）：该策略在我们实测中
被 1e-5 级噪声触发，且衰减后出现过一次整轮崩溃（AUC 0.8566 → 0.6816）。

本脚本自建训练循环，学习率按预设曲线变化，不再依赖单轮指标波动。
早停仍然保留（patience 2），但不再与衰减耦合。
"""
import argparse, json, math, os, sys, time, gc
import numpy as np
import torch
from tqdm import tqdm
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader


def lr_at(step, total, warmup, base_lr, min_ratio):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expid", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--clip-norm", type=float, default=10.0)
    ap.add_argument("--repo", required=True, help="FCN 仓库根，用于 import src")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import src as model_zoo

    params = load_config(a.config, a.expid)
    params["gpu"] = a.gpu
    set_logger(params)
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)

    model = getattr(model_zoo, params["model"])(feature_map, **params)
    model.count_parameters()

    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()
    steps_per_epoch = len(train_gen)
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = int(a.warmup_epochs * steps_per_epoch)
    base_lr = float(params["learning_rate"])
    print(f"[cos] epochs={a.epochs} steps/epoch={steps_per_epoch} warmup={warmup_steps} base_lr={base_lr}")

    best_auc, best_epoch, bad = -1.0, 0, 0
    ckpt = os.path.join(model.model_dir, model.model_id + ".model")
    os.makedirs(model.model_dir, exist_ok=True)

    step = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        for batch in tqdm(train_gen, disable=False, file=sys.stdout):
            lr = lr_at(step, total_steps, warmup_steps, base_lr, a.min_lr_ratio)
            for g in model.optimizer.param_groups:
                g["lr"] = lr
            model.optimizer.zero_grad()
            loss = model.get_total_loss(batch)
            loss.backward()
            if a.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_norm)
            model.optimizer.step()
            step += 1

        res = model.evaluate(valid_gen)
        auc = float(res["AUC"])
        print(f"[cos] Epoch={epoch} lr={lr:.6f} AUC={auc:.6f} logloss={float(res['logloss']):.6f} "
              f"time={time.time()-t0:.1f}s")
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch, 0
            model.save_weights(ckpt)
            print(f"[cos] save best @epoch {epoch}: {auc:.6f}")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[cos] early stop @epoch {epoch} (best {best_auc:.6f} @{best_epoch})")
                break

    model.load_weights(ckpt)
    valid_res = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    test_gen = H5DataLoader(feature_map, stage="test", **params).make_iterator()
    test_res = model.evaluate(test_gen) if test_gen else {}
    print(f"VALID {json.dumps({k: float(v) for k, v in valid_res.items()})}")
    print(f"TEST  {json.dumps({k: float(v) for k, v in test_res.items()})}")
    print(f"BEST_EPOCH {best_epoch}")


if __name__ == "__main__":
    main()
