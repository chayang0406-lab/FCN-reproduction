"""用作者全部可解析日志，逐份验证 param_formula.py 的两个公式。

新版用结构推导公式；旧版用经验拟合式。两者都必须 100% 命中。
这是比 selftest 更强的验证：selftest 的用例是人挑的，这里是把所有日志全跑一遍。
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_formula import count_params, count_params_old_impl

LOG_DIR = os.environ.get(
    "LOG_DIR",
    "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo/checkpoints/Criteo_x4_10_h5")


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "FCNv2_*.log")))
    print("扫描目录:", LOG_DIR)
    print("FCNv2 日志数:", len(files))
    print()

    n_new = n_old = 0
    ok_new = ok_old = 0
    fails = []
    skipped = []

    for p in files:
        name = os.path.basename(p)
        text = open(p, encoding="utf-8", errors="replace").read()
        if not text.strip():
            skipped.append((name, "空文件(0字节)"))
            continue
        m = re.search(r"Total number of parameters: (\d+)", text)
        if not m:
            skipped.append((name, "无参数量行"))
            continue
        actual = int(m.group(1))

        me = re.search(r'"exp_num_layers": "(\d+)"', text)
        ml = re.search(r'"lin_num_layers": "(\d+)"', text)
        if me and ml:
            e, l = int(me.group(1)), int(ml.group(1))
            pred = count_params(e, l)
            kind, label = "NEW", "exp=%d lin=%d" % (e, l)
            n_new += 1
            if pred == actual:
                ok_new += 1
            else:
                fails.append((name, kind, label, pred, actual))
            continue

        me2 = re.search(r'"ECN_num_layers": "(\d+)"', text)
        ml2 = re.search(r'"LCN_num_layers": "(\d+)"', text)
        if me2 and ml2:
            e, l = int(me2.group(1)), int(ml2.group(1))
            pred = count_params_old_impl(e, l)
            kind, label = "OLD", "ECN=%d LCN=%d" % (e, l)
            n_old += 1
            if pred == actual:
                ok_old += 1
            else:
                fails.append((name, kind, label, pred, actual))
            continue

        skipped.append((name, "无层数配置"))

    print("%-42s %-5s %-20s %-11s %-11s %s" % ("log", "kind", "config", "pred", "actual", "结果"))
    print("-" * 104)
    for p in files:
        name = os.path.basename(p)
        text = open(p, encoding="utf-8", errors="replace").read()
        if not text.strip():
            continue
        m = re.search(r"Total number of parameters: (\d+)", text)
        if not m:
            continue
        actual = int(m.group(1))
        me = re.search(r'"exp_num_layers": "(\d+)"', text)
        ml = re.search(r'"lin_num_layers": "(\d+)"', text)
        if me and ml:
            e, l = int(me.group(1)), int(ml.group(1))
            pred = count_params(e, l)
            print("%-42s %-5s %-20s %-11d %-11d %s" % (
                name, "NEW", "exp=%d lin=%d" % (e, l), pred, actual,
                "OK" if pred == actual else "FAIL"))
        else:
            me2 = re.search(r'"ECN_num_layers": "(\d+)"', text)
            ml2 = re.search(r'"LCN_num_layers": "(\d+)"', text)
            if me2 and ml2:
                e, l = int(me2.group(1)), int(ml2.group(1))
                pred = count_params_old_impl(e, l)
                print("%-42s %-5s %-20s %-11d %-11d %s" % (
                    name, "OLD", "ECN=%d LCN=%d" % (e, l), pred, actual,
                    "OK" if pred == actual else "FAIL"))

    print()
    print("=" * 84)
    print("新版(结构推导): %d/%d 命中" % (ok_new, n_new))
    print("旧版(经验拟合): %d/%d 命中" % (ok_old, n_old))
    print("跳过 %d 份:" % len(skipped))
    for n, why in skipped:
        print("   - %-42s %s" % (n, why))
    print()
    total_ok = (ok_new == n_new) and (ok_old == n_old)
    if fails:
        print("FAIL 明细:")
        for f in fails:
            print("   ", f)
        return 1
    print("PASS: 全部 %d 份日志 100%% 命中（新版 %d + 旧版 %d）"
          % (n_new + n_old, n_new, n_old))
    return 0


if __name__ == "__main__":
    sys.exit(main())

