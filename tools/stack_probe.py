"""逐样本门控的天花板：在 valid 上拟合 [logit_d, logit_s] 的线性组合，在 test 上评估。

给出"任何逐样本线性融合"能达到的上界，用于判断门控类改进是否值得做。
"""
import argparse, os, sys, json
import numpy as np, torch
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--expid", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, a.repo)
import src as model_zoo
params = load_config(a.config, a.expid); params["gpu"] = a.gpu
data_dir = os.path.join(params["data_root"], params["dataset_id"])
fm = FeatureMap(params["dataset_id"], data_dir)
fm.load(os.path.join(data_dir, "feature_map.json"), params)
model = getattr(model_zoo, params["model"])(fm, **params)
model.load_weights(os.path.join(params["model_root"], params["dataset_id"], a.expid + ".model"))
model.eval(); model.testing = True

def collect(stage):
    if stage == "valid":
        _, gen = H5DataLoader(fm, stage="train", **params).make_iterator()
    else:
        gen = H5DataLoader(fm, stage="test", **params).make_iterator()
    Pd, Ps, Y = [], [], []
    with torch.no_grad():
        for batch in gen:
            out = model.forward(batch)
            Pd.append(out["y_d"].detach().cpu().numpy().ravel())
            Ps.append(out["y_s"].detach().cpu().numpy().ravel())
            Y.append(model.get_labels(batch).detach().cpu().numpy().ravel())
    return np.concatenate(Pd), np.concatenate(Ps), np.concatenate(Y)

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))

vd, vs, vy = collect("valid"); td, ts, ty = collect("test")
Lv = np.column_stack([logit(vd), logit(vs)]); Lt = np.column_stack([logit(td), logit(ts)])

lr = LogisticRegression(max_iter=1000).fit(Lv, vy)
auc_stack = roc_auc_score(ty, lr.decision_function(Lt))
auc_half = roc_auc_score(ty, 0.5 * Lt[:, 0] + 0.5 * Lt[:, 1])
print("SUMMARY %s | stacking=%.6f  mean(0.5)=%.6f  gain=%+.6f | coef=%s intercept=%.4f" % (
    a.expid, auc_stack, auc_half, auc_stack - auc_half,
    np.round(lr.coef_[0], 4).tolist(), lr.intercept_[0]))
