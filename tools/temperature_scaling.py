"""温度缩放：在拟合集上学一个标量 T，评估测试集校准改善。

p' = sigmoid(logit(p) / T)。T>1 软化（缓解过自信），T<1 锐化。
保序变换，AUC 理论上不变。
"""
import argparse, json, os, sys
import numpy as np, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--fit-stage", default="valid", choices=["valid", "train"])
ap.add_argument("--max-batches", type=int, default=0, help=">0 时只用前 N 个 batch 拟合")
ap.add_argument("--bins", type=int, default=20)
a = ap.parse_args()

sys.path.insert(0, a.repo)
import src as model_zoo
params = load_config(a.config, a.expid); params["gpu"] = a.gpu
d = os.path.join(params["data_root"], params["dataset_id"])
fm = FeatureMap(params["dataset_id"], d); fm.load(os.path.join(d, "feature_map.json"), params)
model = getattr(model_zoo, params["model"])(fm, **params)
model.load_weights(os.path.join(params["model_root"], params["dataset_id"], a.expid + ".model"))
model.eval(); model.testing = True

def collect(stage, max_batches=0):
    if stage == "valid":
        _, gen = H5DataLoader(fm, stage="train", **params).make_iterator()
    else:
        gen = H5DataLoader(fm, stage="test", **params).make_iterator()
    P, Y = [], []
    with torch.no_grad():
        for i, batch in enumerate(gen):
            if max_batches and i >= max_batches: break
            o = model.forward(batch)
            P.append(o["y_pred"].detach().cpu().numpy().ravel())
            Y.append(model.get_labels(batch).detach().cpu().numpy().ravel())
    return np.concatenate(P), np.concatenate(Y)

def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))

def ece_mce(prob, label, bins):
    order = np.argsort(prob, kind="stable"); chunks = np.array_split(order, bins)
    ece = 0.0; mce = 0.0
    for idx in chunks:
        if len(idx) == 0: continue
        gap = abs(float(label[idx].mean()) - float(prob[idx].mean()))
        ece += len(idx) / len(prob) * gap; mce = max(mce, gap)
    return ece, mce

pf, yf = collect(a.fit_stage, a.max_batches)
pt, yt = collect("test")
zf, zt = lg(pf), lg(pt)
clip = lambda z: np.clip(z, -30, 30)

best_T, best_nll = 1.0, float("inf")
for T in np.arange(0.30, 6.01, 0.05):
    q = 1 / (1 + np.exp(-clip(zf / T)))
    nll = -np.mean(yf * np.log(np.clip(q, 1e-9, 1)) + (1 - yf) * np.log(np.clip(1 - q, 1e-9, 1)))
    if nll < best_nll: best_nll, best_T = nll, float(T)

q0 = 1 / (1 + np.exp(-clip(zt)))
q1 = 1 / (1 + np.exp(-clip(zt / best_T)))
e0, m0 = ece_mce(q0, yt, a.bins); e1, m1 = ece_mce(q1, yt, a.bins)
res = {"expid": a.expid, "T": best_T, "fit_stage": a.fit_stage, "n_fit": int(len(yf)), "n_test": int(len(yt)),
       "ece_before": e0, "ece_after": e1, "mce_before": m0, "mce_after": m1,
       "brier_before": float(np.mean((q0 - yt) ** 2)), "brier_after": float(np.mean((q1 - yt) ** 2)),
       "auc_before": float(roc_auc_score(yt, q0)), "auc_after": float(roc_auc_score(yt, q1))}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/ts_{a.expid}.json", "w"), indent=2)
print("SUMMARY %s | T*=%.2f | ECE %.6f -> %.6f (%+.1f%%) | MCE %.6f -> %.6f | Brier %.6f -> %.6f | AUC %.6f -> %.6f" % (
    a.expid, best_T, e0, e1, (e1 - e0) / max(e0, 1e-12) * 100, m0, m1,
    res["brier_before"], res["brier_after"], res["auc_before"], res["auc_after"]))
