"""推理期融合权重扫描：不训练，用已训模型评估 alpha 对 AUC 的影响。

alpha=0.5 应复现原模型指标（logit 空间取平均，与 FCNv1.forward 一致）。
"""
import argparse, os, sys, json
import numpy as np, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--step", type=float, default=0.025)
a = ap.parse_args()

sys.path.insert(0, a.repo)
import src as model_zoo

params = load_config(a.config, a.expid)
params["gpu"] = a.gpu
data_dir = os.path.join(params["data_root"], params["dataset_id"])
fm = FeatureMap(params["dataset_id"], data_dir)
fm.load(os.path.join(data_dir, "feature_map.json"), params)

model = getattr(model_zoo, params["model"])(fm, **params)
wpath = os.path.join(params["model_root"], params["dataset_id"], a.expid + ".model")
model.load_weights(wpath)
model.eval()
model.testing = True

gen = H5DataLoader(fm, stage="test", **params).make_iterator()
Pd, Ps, Ys, Ypred = [], [], [], []
with torch.no_grad():
    for batch in gen:
        out = model.forward(batch)
        y = model.get_labels(batch)
        Pd.append(out["y_d"].detach().cpu().numpy().ravel())
        Ps.append(out["y_s"].detach().cpu().numpy().ravel())
        Ypred.append(out["y_pred"].detach().cpu().numpy().ravel())
        Ys.append(y.detach().cpu().numpy().ravel())

pd_ = np.concatenate(Pd); ps_ = np.concatenate(Ps); y_ = np.concatenate(Ys); yp_ = np.concatenate(Ypred)
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
ld, ls = logit(pd_), logit(ps_)

rows = []
for alpha in np.arange(0.0, 1.0001, a.step):
    auc = roc_auc_score(y_, alpha * ld + (1 - alpha) * ls)
    rows.append((round(float(alpha), 4), float(auc)))

base = dict((r[0], r[1]) for r in rows)[0.5]
best = max(rows, key=lambda r: r[1])
res = {
    "expid": a.expid,
    "n_test": int(len(y_)),
    "auc_ecn_only(alpha=1)": round(rows[-1][1], 6),
    "auc_lcn_only(alpha=0)": round(rows[0][1], 6),
    "auc_alpha_0.5": round(base, 6),
    "auc_model_y_pred": round(float(roc_auc_score(y_, yp_)), 6),
    "best_alpha": best[0],
    "best_auc": round(best[1], 6),
    "gain_vs_0.5": round(best[1] - base, 6),
    "curve": rows,
}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/fusion_{a.expid}.json", "w"), ensure_ascii=False, indent=2)
print("SUMMARY", a.expid, "| alpha*=%.3f  auc*=%.6f  auc@0.5=%.6f  gain=%+.6f  (ECN only=%.6f, LCN only=%.6f)" % (
    res["best_alpha"], res["best_auc"], res["auc_alpha_0.5"], res["gain_vs_0.5"],
    res["auc_ecn_only(alpha=1)"], res["auc_lcn_only(alpha=0)"]))
