"""机制诊断：Tri-BCE 权重激活率、分支相关性、损失分解。

不重新训练，只用已保存模型在测试集上前向。
"""
import argparse, os, sys, json
import numpy as np, torch
import torch.nn.functional as F
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.repo)
import src as model_zoo

params = load_config(a.config, a.expid); params["gpu"] = a.gpu
d = os.path.join(params["data_root"], params["dataset_id"])
fm = FeatureMap(params["dataset_id"], d); fm.load(os.path.join(d, "feature_map.json"), params)
model = getattr(model_zoo, params["model"])(fm, **params)
model.load_weights(os.path.join(params["model_root"], params["dataset_id"], a.expid + ".model"))
model.eval(); model.testing = True

gen = H5DataLoader(fm, stage="test", **params).make_iterator()
P, PD, PS, Y = [], [], [], []
with torch.no_grad():
    for batch in gen:
        o = model.forward(batch)
        P.append(o["y_pred"].detach().cpu().numpy().ravel())
        PD.append(o["y_d"].detach().cpu().numpy().ravel())
        PS.append(o["y_s"].detach().cpu().numpy().ravel())
        Y.append(model.get_labels(batch).detach().cpu().numpy().ravel())
p, pd_, ps_, y = map(np.concatenate, (P, PD, PS, Y))

def bce(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))

L, LD, LS = bce(p, y), bce(pd_, y), bce(ps_, y)
wD = np.maximum(LD - L, 0); wS = np.maximum(LS - L, 0)

def logit(x):
    x = np.clip(x, 1e-7, 1 - 1e-7); return np.log(x / (1 - x))
corr = float(np.corrcoef(logit(pd_), logit(ps_))[0, 1])

out = {
    "expid": a.expid, "n": int(len(y)),
    "loss_fused": float(L.mean()), "loss_ecn": float(LD.mean()), "loss_lcn": float(LS.mean()),
    "wD_active_rate": float((wD > 0).mean()), "wS_active_rate": float((wS > 0).mean()),
    "wD_mean_active": float(wD[wD > 0].mean()) if (wD > 0).any() else 0.0,
    "wS_mean_active": float(wS[wS > 0].mean()) if (wS > 0).any() else 0.0,
    "wD_max": float(wD.max()), "wS_max": float(wS.max()),
    "branch_logit_corr": corr,
    "pos_rate": float(y.mean()),
}
print("SUMMARY %s | L=%.6f LD=%.6f LS=%.6f | wD_active=%.2f%% wS_active=%.2f%% | wD_mean=%.6f wS_mean=%.6f | corr=%.4f" % (
    a.expid, out["loss_fused"], out["loss_ecn"], out["loss_lcn"],
    out["wD_active_rate"]*100, out["wS_active_rate"]*100,
    out["wD_mean_active"], out["wS_mean_active"], corr))
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/mech_{a.expid}.json", "w"), indent=2)
