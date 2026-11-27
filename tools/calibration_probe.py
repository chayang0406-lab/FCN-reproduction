"""CTR 概率校准探针：ECE / MCE / Brier（不重新训练）。

按等频（分位数）分箱——iPinYou 正例率约 0.07%，等宽分箱会让几乎所有样本落在
第一个箱里、ECE 失真；等频分箱才能看出"高分区是否过/欠自信"。
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

def ece_mce(prob, label, bins):
    order = np.argsort(prob, kind="stable")
    chunks = np.array_split(order, bins)
    ece = 0.0; mce = 0.0; rows = []
    for idx in chunks:
        if len(idx) == 0: continue
        conf = float(prob[idx].mean()); acc = float(label[idx].mean()); w = len(idx) / len(prob)
        gap = abs(acc - conf)
        ece += w * gap; mce = max(mce, gap)
        rows.append({"n": int(len(idx)), "conf": round(conf, 6), "acc": round(acc, 6), "gap": round(gap, 6)})
    return ece, mce, rows

out = {"expid": a.expid, "n": int(len(y)), "pos_rate": float(y.mean()), "methods": {}}
for name, prob in (("fused", p), ("ecn_only", pd_), ("lcn_only", ps_)):
    ece, mce, rows = ece_mce(prob, y, a.bins)
    out["methods"][name] = {
        "auc": float(roc_auc_score(y, prob)),
        "brier": float(np.mean((prob - y) ** 2)),
        "mean_pred": float(prob.mean()),
        "ece": ece, "mce": mce, "reliability": rows,
    }
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/calib_{a.expid}.json", "w"), indent=2)

m = out["methods"]
print("SUMMARY %s | pos=%.5f%% | fused ECE=%.5f MCE=%.5f Brier=%.6f | ecn ECE=%.5f | lcn ECE=%.5f | AUC fused=%.6f" % (
    a.expid, out["pos_rate"]*100, m["fused"]["ece"], m["fused"]["mce"], m["fused"]["brier"],
    m["ecn_only"]["ece"], m["lcn_only"]["ece"], m["fused"]["auc"]))
print("MEANPRED fused=%.6f vs pos=%.6f (ratio %.2fx)" % (m["fused"]["mean_pred"], out["pos_rate"], m["fused"]["mean_pred"]/max(out["pos_rate"],1e-12)))
