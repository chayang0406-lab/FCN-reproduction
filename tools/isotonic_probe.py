"""事后校准对比：温度缩放 vs 等渗回归（isotonic）。

等渗回归是分段常数的单调映射；但样本少的分箱会过拟合。
在同一拟合集上训练两种校准器，在测试集上比较 ECE / Brier / AUC。
"""
import argparse, json, os, sys
import numpy as np, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--fit-stage", default="valid", choices=["valid", "train"])
ap.add_argument("--max-batches", type=int, default=0)
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

def collect(stage, mb=0):
    if stage == "valid":
        _, gen = H5DataLoader(fm, stage="train", **params).make_iterator()
    else:
        gen = H5DataLoader(fm, stage="test", **params).make_iterator()
    P, Y = [], []
    with torch.no_grad():
        for i, b in enumerate(gen):
            if mb and i >= mb: break
            o = model.forward(b)
            P.append(o["y_pred"].detach().cpu().numpy().ravel())
            Y.append(model.get_labels(b).detach().cpu().numpy().ravel())
    return np.concatenate(P), np.concatenate(Y)

def ece(prob, label, bins):
    o = np.argsort(prob, kind="stable"); chunks = np.array_split(o, bins)
    return float(sum(len(i)/len(prob)*abs(float(label[i].mean())-float(prob[i].mean())) for i in chunks if len(i)))

pf, yf = collect(a.fit_stage, a.max_batches); pt, yt = collect("test")
def lg(p):
    p = np.clip(p, 1e-9, 1-1e-9); return np.log(p/(1-p))
zf, zt = lg(pf), lg(pt)

best_T, best_nll = 1.0, 1e18
for T in np.arange(0.3, 6.01, 0.05):
    q = 1/(1+np.exp(-np.clip(zf/T, -30, 30)))
    nll = -np.mean(yf*np.log(np.clip(q,1e-9,1)) + (1-yf)*np.log(np.clip(1-q,1e-9,1)))
    if nll < best_nll: best_nll, best_T = nll, T

iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-7, y_max=1-1e-7).fit(pf, yf)
q_ts = 1/(1+np.exp(-np.clip(zt/best_T, -30, 30)))
q_iso = np.clip(iso.predict(pt), 1e-7, 1-1e-7)

res = {"expid": a.expid, "T": best_T, "T_ece": ece(q_ts, yt, a.bins), "iso_ece": ece(q_iso, yt, a.bins),
       "base_ece": ece(pt, yt, a.bins), "T_brier": float(np.mean((q_ts-yt)**2)), "iso_brier": float(np.mean((q_iso-yt)**2)),
       "base_brier": float(np.mean((pt-yt)**2)),
       "T_auc": float(roc_auc_score(yt, q_ts)), "iso_auc": float(roc_auc_score(yt, q_iso)), "base_auc": float(roc_auc_score(yt, pt))}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/posthoc_{a.expid}.json","w"), indent=2)
print("SUMMARY %s | base ECE %.6f Brier %.6f AUC %.6f | TS(T=%.2f) ECE %.6f Brier %.6f AUC %.6f | ISO ECE %.6f Brier %.6f AUC %.6f" % (
    a.expid, res["base_ece"], res["base_brier"], res["base_auc"], best_T, res["T_ece"], res["T_brier"], res["T_auc"],
    res["iso_ece"], res["iso_brier"], res["iso_auc"]))
