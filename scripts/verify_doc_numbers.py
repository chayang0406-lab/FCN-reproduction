"""最终校验：文档里引用的每一个 AUC 数字都必须能从原始日志逐字复现。

这个脚本存在的意义：上一版审计脚本曾把 test AUC 当 valid AUC，
导致文档里出现错误标注。现在用「直接从原始文本抓取，不做任何推断」的方式复核，
确保文档写的每一个数都能在日志里原文找到。
"""
import glob
import os
import re
import sys

LOG_DIR = os.environ.get("LOG_DIR", "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/code/FCN_repo/checkpoints/Criteo_x4_10_h5")

# 文档里声明的数字（key = 日志文件名去 .log）
CLAIMS = {
    "FCNv2_Criteo": {"valid": 0.815699, "test": 0.816099, "params": 16138194},
    "FCNv2_Criteo_016_cde6d2ff": {"valid": 0.815627, "test": 0.816145, "params": 16532562},
    "FCNv2_Criteo_017_30a838c6": {"valid": 0.815818, "test": 0.816352, "params": 16924434},
    "FCNv2_Criteo_025_747baf17": {"valid": 0.815861, "test": 0.816318, "params": 18491922},
}


def extract(path):
    """直接读原文，用「分段位置」定位 valid 与 test，不做任何猜测。"""
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    if not lines:
        return None

    i_val = i_test = None
    epoch_aucs = []
    params = None

    for i, ln in enumerate(lines):
        if "Total number of parameters:" in ln:
            m = re.search(r"(\d+)", ln.split(":")[-1])
            if m:
                params = int(m.group(1))
        if "Validation evaluation" in ln:
            i_val = i
        elif "Test evaluation" in ln:
            i_test = i
        elif "[Metrics]" in ln and i_val is None and i_test is None and "AUC" in ln:
            m = re.search(r"AUC: (0\.\d+)", ln)
            if m:
                epoch_aucs.append(float(m.group(1)))

    def grab(idx):
        if idx is None:
            return None
        for j in range(idx + 1, min(idx + 12, len(lines))):
            m = re.search(r"AUC: (0\.\d+)", lines[j])
            if m:
                return float(m.group(1))
        return None

    return {
        "valid_epoch_best": max(epoch_aucs) if epoch_aucs else None,
        "valid_final": grab(i_val),
        "test": grab(i_test),
        "params": params,
    }


def main():
    fails = []
    print("%-32s %-12s %-12s %-12s %s" % ("log", "claim(valid)", "actual", "claim(test)", "actual"))
    print("-" * 90)
    for stem, claim in CLAIMS.items():
        p = os.path.join(LOG_DIR, stem + ".log")
        if not os.path.exists(p):
            print("%-32s MISSING FILE" % stem)
            fails.append(stem + " (missing)")
            continue
        got = extract(p)
        if got is None:
            print("%-32s EMPTY FILE" % stem)
            fails.append(stem + " (empty)")
            continue

        ok_v = got["valid_final"] == claim["valid"]
        ok_t = got["test"] == claim["test"]
        ok_p = got["params"] == claim["params"]
        print("%-32s %-12s %-12s %-12s %s" % (
            stem,
            "%.6f" % claim["valid"], "%.6f %s" % (got["valid_final"], "OK" if ok_v else "MISMATCH"),
            "%.6f" % claim["test"], "%.6f %s" % (got["test"], "OK" if ok_t else "MISMATCH")))
        if not ok_v:
            fails.append("%s valid: claim %.6f vs actual %s" % (stem, claim["valid"], got["valid_final"]))
        if not ok_t:
            fails.append("%s test: claim %.6f vs actual %s" % (stem, claim["test"], got["test"]))
        if not ok_p:
            fails.append("%s params: claim %d vs actual %s" % (stem, claim["params"], got["params"]))

    print()
    print("=== valid 与 test 必须不同（若相等说明分段解析失败）===")
    for stem in CLAIMS:
        got = extract(os.path.join(LOG_DIR, stem + ".log"))
        if got and got["valid_final"] and got["test"]:
            same = got["valid_final"] == got["test"]
            print("  %-32s valid=%s test=%s  %s" % (
                stem, got["valid_final"], got["test"], "!! 相同，可疑" if same else "差异正常"))
            if same:
                fails.append(stem + " valid==test suspicious")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL CLAIMS VERIFIED against raw logs (%d logs)" % len(CLAIMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

