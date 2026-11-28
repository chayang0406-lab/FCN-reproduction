"""校准的时段稳定性（统计检验版）。

关键：iPinYou 正例率仅 0.073%，每小时约 20 万条，若按 20 箱切分，
每箱只有约 7 个正例，ECE 的抽样噪声可达几十个百分点。因此必须先做显著性检验。

本脚本对每个分组计算：
  CITL = mean(p) - mean(y)                   整体偏差
  z    = CITL / sqrt(SE_p^2 + SE_y^2)        是否显著偏离 0
  RCITL = CITL / mean(y)                     相对偏差
只有 |z| > 2 的分组才视为存在真实失准。
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
ap.add_argument("--group-field", default="hour")
ap.add_argument("--min-n", type=int, default=20000)
a = ap.parse_args()
sys.path.insert(0, a.repo)
import src as model_zoo
params = load_config(a.config, a.expid); params["gpu"] = a.gpu
d = os.path.join(params["data_root"], params["dataset_id"])
fm = FeatureMap(params["dataset_id"], d); fm.load(os.path.join(d, "feature_map.json"), params)
model = getattr(model_zoo, params["model"])(fm, **params)
model.load_weights(os.path.join(params["model_root"], params["dataset_id"], a.expid + ".model"))
model.eval(); model.testing = True
col = fm.get_column_index(a.group_field)

gen = H5DataLoader(fm, stage="test", **params).make_iterator()
P, Y, G = [], [], []
with torch.no_grad():
    for b in gen:
        o = model.forward(b)
        P.append(o["y_pred"].detach().cpu().numpy().ravel())
        Y.append(model.get_labels(b).detach().cpu().numpy().ravel())
        G.append(b[:, col].detach().cpu().numpy().astype("int64"))
p = np.concatenate(P).astype(np.float64); y = np.concatenate(Y).astype(np.float64)
g = np.concatenate(G)

rows = []
for v in np.unique(g):
    m = g == v
    n = int(m.sum())
    if n < a.min_n: continue
    pv, yv = p[m], y[m]
    mean_p, mean_y = float(pv.mean()), float(yv.mean())
    se_y = float(np.sqrt(max(mean_y * (1 - mean_y), 1e-18) / n))
    se_p = float(np.sqrt(max(pv.var(), 1e-18) / n))
    citl = mean_p - mean_y
    z = citl / np.sqrt(se_y**2 + se_p**2) if (se_y or se_p) else 0.0
    rows.append({"group": int(v), "n": n, "base_rate": mean_y, "mean_pred": mean_p,
                 "citl": citl, "citl_rel": citl / mean_y if mean_y > 0 else None,
                 "z": float(z), "auc": float(roc_auc_score(yv, pv))})

os.makedirs("results", exist_ok=True)
json.dump({"expid": a.expid, "field": a.group_field, "rows": rows},
          open(f"results/temporal_{a.expid}_{a.group_field}.json", "w"), indent=2)
sig = [r for r in rows if abs(r["z"]) > 2]
print("SUMMARY %s by %s | 组数=%d | 显著失准组(|z|>2)=%d" % (a.expid, a.group_field, len(rows), len(sig)))
print("  %-6s %-9s %-9s %-9s %-8s %-8s" % ("group", "n", "base%", "RCITL", "z", "AUC"))
for r in rows:
    print("  %-6d %-9d %-9.4f %-9.2f %-8.2f %-8.4f" % (
        r["group"], r["n"], r["base_rate"]*100, (r["citl_rel"] or 0)*100, r["z"], r["auc"]))
rels = [r["citl_rel"] for r in rows if r["citl_rel"] is not None]
print("  RCITL: 中位=%+.2f%% 极差=%.2f%% | 显著组中最大|RCITL|=%.2f%%" % (
    float(np.median(rels))*100, (max(rels)-min(rels))*100,
    max([abs(r["citl_rel"]) for r in sig], default=0)*100))
