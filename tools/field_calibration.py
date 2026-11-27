"""分字段校准诊断（pandas 分组聚合版）。

对每个字段按取值分组，统计 (平均预测 - 实际点击率)，按 |gap|*n 排序，
并用 sum(|gap|*n/N) 近似该字段对全局 ECE 的贡献。
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--fields", default="")
ap.add_argument("--topk", type=int, default=3)
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
P, Y, X = [], [], []
with torch.no_grad():
    for b in gen:
        o = model.forward(b)
        P.append(o["y_pred"].detach().cpu().numpy().ravel())
        Y.append(model.get_labels(b).detach().cpu().numpy().ravel())
        X.append(b.detach().cpu().numpy().astype("int64"))
p, y, Xa = np.concatenate(P), np.concatenate(Y), np.concatenate(X)

fields = [f.strip() for f in a.fields.split(",") if f.strip()] or list(fm.features.keys())
df = pd.DataFrame({"p": p, "y": y})
out = {"expid": a.expid, "n": int(len(y)), "global_gap": float(p.mean() - y.mean()), "fields": {}}
for f in fields:
    try:
        idx = fm.get_column_index(f)
    except Exception:
        continue
    df["g"] = Xa[:, idx]
    agg = df.groupby("g", sort=False).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    agg = agg[agg["n"] >= 500].copy()
    if agg.empty: continue
    agg["gap"] = agg["pred"] - agg["actual"]
    agg["contrib"] = agg["gap"].abs() * agg["n"] / len(df)
    agg = agg.sort_values("contrib", ascending=False)
    out["fields"][f] = {
        "n_groups": int(len(agg)),
        "field_contrib": float(agg["contrib"].sum()),
        "worst": [{"value": int(v), "n": int(r.n), "pred": float(r.pred), "actual": float(r.actual), "gap": float(r.gap)}
                  for v, r in agg.head(a.topk).iterrows()],
    }
del df
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/fieldcalib_{a.expid}.json", "w"), indent=2)
print(f"SUMMARY {a.expid} | global pred-actual = {out['global_gap']:+.6f}")
for f, v in sorted(out["fields"].items(), key=lambda kv: -kv[1]["field_contrib"])[:5]:
    w = v["worst"][0]
    print(f"  {f:20s} contrib={v['field_contrib']:.6f} worst: value={w['value']} n={w['n']} pred={w['pred']:.5f} act={w['actual']:.5f} gap={w['gap']:+.5f}")
