import argparse, gc, json, logging, os, sys, time
import numpy as np
if not hasattr(np, "Inf"): np.Inf = np.inf

IPINYOU = {
    "name": "iPinYou",
    "label_col": {"dtype": "float", "name": "click"},
    "min_categr_count": 2,
    "feature_cols": [
        {"active": True, "dtype": "str", "type": "categorical",
         "name": ["weekday", "hour", "IP", "region", "city", "adexchange", "domain",
                  "slotid", "slotwidth", "slotheight", "slotvisibility", "slotformat",
                  "creative", "advertiser", "useragent", "slotprice"]},
    ],
    "processor": "fuxictr.preprocess",
    "vocab": 665765, "fields": 16, "emb": 16,
}

KKBOX = {
    "name": "KKBox_x1",
    "label_col": {"dtype": "float", "name": "label"},
    "min_categr_count": 10,
    "feature_cols": [
        {"active": True, "dtype": "str", "type": "categorical",
         "name": ["msno", "song_id", "source_system_tab", "source_screen_name",
                  "source_type", "city", "gender", "registered_via", "language"]},
        {"active": True, "dtype": "str", "type": "sequence",
         "encoder": "MaskedSumPooling", "max_len": 3, "name": "genre_ids"},
        {"active": True, "dtype": "str", "type": "sequence",
         "encoder": "MaskedSumPooling", "max_len": 3, "name": "artist_name"},
        {"active": True, "dtype": "str", "type": "categorical", "name": "isrc",
         "preprocess": "extract_country_code"},
        {"active": True, "dtype": "str", "type": "categorical", "name": "bd",
         "preprocess": "bucketize_age"},
    ],
    "processor": "fuxictr.datasets.kkbox",
    "vocab": 91756, "fields": 13, "emb": 128,
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["ipinyou", "kkbox"])
ap.add_argument("--csv-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset-id", default=None)
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(message)s")
spec = IPINYOU if a.dataset == "ipinyou" else KKBOX
dsid = a.dataset_id or (a.dataset + "_x1_h5")
csvd = a.csv_dir.rstrip("/") + "/"

def pick(name):
    p = csvd + name
    return p if os.path.isfile(p) else None

train = pick("train.csv")
test = pick("test.csv")
valid = pick("valid.csv") or test
assert train and test, "train.csv / test.csv 缺失，检查 --csv-dir: " + csvd

p = dict(data_format="csv", data_root=os.path.dirname(csvd.rstrip("/")) + "/",
         dataset_id=dsid, train_data=train, valid_data=valid, test_data=test,
         label_col=spec["label_col"], min_categr_count=spec["min_categr_count"],
         feature_cols=spec["feature_cols"], data_block_size=0, verbose=0)

print("train =", train)
print("valid =", valid)
print("test  =", test)
print("processor =", spec["processor"])

FP = getattr(__import__(spec["processor"], fromlist=["FeatureProcessor"]), "FeatureProcessor")
from fuxictr.features import FeatureMap
from fuxictr.preprocess import build_dataset

os.makedirs(a.out, exist_ok=True)
fmj = os.path.join(a.out, "feature_map.json")
if os.path.exists(fmj): os.remove(fmj)

t0 = time.time()
fe = FP(**p)
p["train_data"], p["valid_data"], p["test_data"] = build_dataset(fe, **p)
print("转换完成，耗时 %.1f 分钟" % ((time.time() - t0) / 60))
gc.collect()

fm = FeatureMap(dsid, a.out)
fm.load(fmj, {"embedding_dim": spec["emb"]})
got = {k: v["vocab_size"] for k, v in fm.features.items()}
tot, nf = int(sum(got.values())), int(fm.num_fields)
ok = (tot == spec["vocab"]) and (nf == spec["fields"])

print("=" * 56)
print("vocab_sum = %d  (期望 %d)  %s" % (tot, spec["vocab"], "PASS" if tot == spec["vocab"] else "FAIL"))
print("n_fields  = %d  (期望 %d)  %s" % (nf, spec["fields"], "PASS" if nf == spec["fields"] else "FAIL"))
print("=" * 56)

json.dump({"dataset_id": dsid, "vocab_sum": tot, "n_fields": nf, "vocab_all": got},
          open(os.path.join(a.out, "build_summary.json"), "w"), indent=2, ensure_ascii=False)

for f in sorted(os.listdir(a.out)):
    fp = os.path.join(a.out, f)
    if os.path.isfile(fp):
        print("  %-26s %8.1f MB" % (f, os.path.getsize(fp) / 1e6))
print("PASS" if ok else "FAIL 校验未通过")
sys.exit(0 if ok else 1)
