"""融合前分支校准（branch-wise temperature scaling）。

发现：融合 ECE 夹在两个分支之间——σ(0.5z_d+0.5z_s) 是两分支概率的平均，
平均不可能优于更好的那一支。因此"先融合再校准"是治标。

本方法：先分别给两个分支各自的温度，再融合：
    p = sigmoid( 0.5*z_d/T_d + 0.5*z_s/T_s )
令 T_d = T_s = T 即退化为常规的事后温度缩放，因此本方法严格包含基线。
"""
import argparse, json, os, sys
import numpy as np, torch
from scipy.optimize import minimize
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
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
    PD, PS, Y = [], [], []
    with torch.no_grad():
        for i, b in enumerate(gen):
            if mb and i >= mb: break
            o = model.forward(b)
            PD.append(o["y_d"].detach().cpu().numpy().ravel())
            PS.append(o["y_s"].detach().cpu().numpy().ravel())
            Y.append(model.get_labels(b).detach().cpu().numpy().ravel())
    return np.concatenate(PD), np.concatenate(PS), np.concatenate(Y)

def lg(p):
    # 必须转 float64：float32 下 1-1e-12 会被舍入成 1.0，导致 log(0) = -inf
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def ece_mce(prob, label, bins):
    o = np.argsort(prob, kind="stable"); chunks = np.array_split(o, bins)
    e = 0.0; m = 0.0
    for idx in chunks:
        if len(idx) == 0: continue
        g = abs(float(label[idx].mean()) - float(prob[idx].mean()))
        e += len(idx) / len(prob) * g; m = max(m, g)
    return e, m
def nll(q, y):
    q = np.asarray(q, dtype=np.float64)
    # 两项分别裁剪：饱和到 1.0 时 1-q 恰为 0，整体裁剪救不回来
    return -np.mean(y * np.log(np.clip(q, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1.0)))
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

pd_f, ps_f, yf = collect(a.fit_stage, a.max_batches)
pd_t, ps_t, yt = collect("test")
zd_f, zs_f = lg(pd_f), lg(ps_f)
zd_t, zs_t = lg(pd_t), lg(ps_t)

# 1) 融合后单一温度（基线方法）
r1 = minimize(lambda v: nll(sig((0.5*zd_f + 0.5*zs_f) / np.exp(v[0])), yf), [0.0], method="Nelder-Mead",
              options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 500})
T_glob = float(np.exp(r1.x[0]))
# 2) 分支级两个温度（本方法）
r2 = minimize(lambda v: nll(sig(0.5*zd_f/np.exp(v[0]) + 0.5*zs_f/np.exp(v[1])), yf), [0.0, 0.0],
              method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 2000})
Td, Ts = float(np.exp(r2.x[0])), float(np.exp(r2.x[1]))

base = sig(0.5*zd_t + 0.5*zs_t)
q_glob = sig((0.5*zd_t + 0.5*zs_t) / T_glob)
q_br = sig(0.5*zd_t/Td + 0.5*zs_t/Ts)
# 3) 分支各自先校准再平均
q_sep = 0.5*sig(zd_t/Td) + 0.5*sig(zs_t/Ts)

out = {"expid": a.expid, "T_global": T_glob, "T_d": Td, "T_s": Ts}
for name, q in (("base", base), ("post_fusion_T", q_glob), ("branch_T_logit", q_br), ("branch_T_probavg", q_sep)):
    e, m = ece_mce(q, yt, a.bins)
    out[name] = {"ece": e, "mce": m, "brier": float(np.mean((q-yt)**2)), "auc": float(roc_auc_score(yt, q))}
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/branchcalib_{a.expid}.json", "w"), indent=2)
print("SUMMARY %s | T_global=%.3f T_d=%.3f T_s=%.3f" % (a.expid, T_glob, Td, Ts))
for k in ("base", "post_fusion_T", "branch_T_logit", "branch_T_probavg"):
    v = out[k]
    print("  %-18s ECE=%.6f MCE=%.6f Brier=%.6f AUC=%.6f" % (k, v["ece"], v["mce"], v["brier"], v["auc"]))
