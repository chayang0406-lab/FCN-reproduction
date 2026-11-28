"""相对校准误差：绝对 ECE 在极端不平衡数据上会低估问题。

ECE 是"概率的绝对差"，但同样的绝对差在不同基准率下含义完全不同。
本脚本同时报告绝对与相对两种口径：
  CITL  = mean(p) - mean(y)          校准偏差（整体）
  ECE   = Σ (n_b/N)|acc_b - conf_b|  期望校准误差（逐箱）
  RCE   = ECE / mean(y)              相对期望校准误差
对竞价场景，RCE 直接对应"出价系统性高估/低估的百分比"。
"""
import argparse, json, os, sys
import numpy as np, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

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
    elif stage == "train":
        gen, _ = H5DataLoader(fm, stage="train", **params).make_iterator()
    else:
        gen = H5DataLoader(fm, stage="test", **params).make_iterator()
    P, Y = [], []
    with torch.no_grad():
        for i, b in enumerate(gen):
            if mb and i >= mb: break
            o = model.forward(b)
            P.append(o["y_pred"].detach().cpu().numpy().ravel())
            Y.append(model.get_labels(b).detach().cpu().numpy().ravel())
    return np.concatenate(P).astype(np.float64), np.concatenate(Y).astype(np.float64)

def bins_of(prob, label, bins):
    o = np.argsort(prob, kind="stable"); out = []
    for idx in np.array_split(o, bins):
        if len(idx) == 0: continue
        out.append((len(idx), float(prob[idx].mean()), float(label[idx].mean())))
    return out

def metrics(prob, label, bins):
    base = float(label.mean()); meanp = float(prob.mean())
    rows = bins_of(prob, label, bins); N = len(prob)
    ece = sum(n/N * abs(acc - conf) for n, conf, acc in rows)
    mce = max(abs(acc - conf) for n, conf, acc in rows)
    return {"base_rate": base, "mean_pred": meanp, "citl": meanp - base,
            "citl_rel": (meanp - base) / base if base > 0 else float("nan"),
            "ece": ece, "mce": mce,
            "rce": ece / base if base > 0 else float("nan"),
            "rmce": mce / base if base > 0 else float("nan"),
            "auc": float(roc_auc_score(label, prob))}

pt, yt = collect("test")
pf, yf = collect(a.fit_stage, a.max_batches)
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-7, y_max=1-1e-7).fit(pf, yf)
pt_iso = np.clip(iso.predict(pt), 1e-7, 1-1e-7)

res = {"expid": a.expid, "n": int(len(yt)),
       "raw": metrics(pt, yt, a.bins), "isotonic": metrics(pt_iso, yt, a.bins)}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/relcalib_{a.expid}.json", "w"), indent=2)
r, i = res["raw"], res["isotonic"]
print("SUMMARY %s | n=%d base=%.6f" % (a.expid, res["n"], r["base_rate"]))
print("  raw      : ECE=%.6f (RCE=%.2f%%)  MCE=%.6f  CITL=%+.6f (%.2f%%)  AUC=%.6f" % (
    r["ece"], r["rce"]*100, r["mce"], r["citl"], r["citl_rel"]*100, r["auc"]))
print("  isotonic : ECE=%.6f (RCE=%.2f%%)  MCE=%.6f  AUC=%.6f" % (
    i["ece"], i["rce"]*100, i["mce"], i["auc"]))
