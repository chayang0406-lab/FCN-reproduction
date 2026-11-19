import importlib
import traceback

mods = [
    "fuxictr",
    "fuxictr.features",
    "fuxictr.utils",
    "fuxictr.preprocess",
    "fuxictr.preprocess.build_dataset",
    "fuxictr.preprocess.feature_processor",
    "fuxictr.pytorch.torch_utils",
    "fuxictr.pytorch.dataloaders",
    "fuxictr.pytorch.layers",
    "fuxictr.pytorch.models",
    "fuxictr.metrics",
    "fuxictr.datasets.criteo",
    "fuxictr.datasets.avazu",
    "fuxictr.datasets.kkbox",
]

ok = 0
fail = []
for m in mods:
    try:
        importlib.import_module(m)
        print("OK   ", m)
        ok += 1
    except Exception as e:
        print("FAIL ", m, "->", type(e).__name__, e)
        fail.append(m)

print()
print("import ok = %d / %d" % (ok, len(mods)))

# 验证 FeatureProcessor 三个版本的能力差异（run_expid.py 的坑）
print()
print("=== FeatureProcessor 能力对照 ===")
from fuxictr.preprocess.feature_processor import FeatureProcessor as Base
from fuxictr.datasets.criteo import FeatureProcessor as Criteo
from fuxictr.datasets.kkbox import FeatureProcessor as Kkbox
from fuxictr.datasets.avazu import FeatureProcessor as Avazu
for name, cls in [("preprocess(通用)", Base), ("datasets.criteo", Criteo),
                  ("datasets.kkbox", Kkbox), ("datasets.avazu", Avazu)]:
    has = hasattr(cls, "convert_to_bucket")
    print("  %-20s convert_to_bucket = %s" % (name, has))

# 验证 src 包（FCN 的 5 个模型文件）全部可导入
print()
print("=== FCN src 包导入 ===")
import sys
sys.path.insert(0, "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo")
import src as model_zoo
names = [n for n in dir(model_zoo) if not n.startswith("_")]
print("  src 暴露的符号:", names)
for n in names:
    obj = getattr(model_zoo, n)
    if isinstance(obj, type):
        print("    模型类:", n, "->", obj.__module__)

print()
print("RESULT:", "PASS" if not fail else "FAIL")
if fail:
    raise SystemExit(1)

