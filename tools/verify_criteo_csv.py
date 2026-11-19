"""校验 Criteo_x4 CSV 三份数据的行数 / 列数 / md5，可选在远程或本地运行。

用法（在数据目录下运行）：
    python3 verify_criteo_csv.py --dir /path/to/data
    python3 verify_criteo_csv.py --dir /path/to/data --json out.json

设计要点：行数是**硬断言**，md5 是**参考断言**。
这两个不等价的理由：官方 md5 对应 reczoo 用 sklearn 0.19.1 切分、pandas 默认
to_csv 写出的文件；如果解压工具或换行归一化不同，md5 会变但数据内容不变。
所以 md5 不符时先看行数+列数，不要直接判定数据损坏。
"""

import argparse
import csv
import hashlib
import json
import os
import sys

EXPECTED = {
    "train.csv": {"rows": 36672493, "md5": "4a53bb7cbc0e4ee25f9d6a73ed824b1a"},
    "valid.csv": {"rows": 4584062, "md5": "fba5428b22895016e790e2dec623cb56"},
    "test.csv": {"rows": 4584062, "md5": "cfc37da0d75c4d2d8778e76997df2976"},
}
EXPECTED_COLS = 40  # label + 13 个 I 字段 + 26 个 C 字段


def md5_file(path, bufsize=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def inspect(path):
    """一次流式扫描，同时得到行数、列数、字段名。"""
    with open(path, "rb") as f:
        head = f.readline()
    crlf = head.endswith(b"\r\n")
    header = head.rstrip(b"\r\n").decode("utf-8", "replace")
    ncols = len(header.split(","))

    rows = 0
    bad = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for rec in reader:
            rows += 1
            if len(rec) != ncols:
                bad += 1
    return {"ncols": ncols, "rows": rows, "bad_rows": bad,
            "crlf": crlf, "header_first3": header.split(",")[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args()

    out = {"dir": os.path.abspath(a.dir), "files": {}, "verdict": None}
    all_ok = True

    for name, exp in EXPECTED.items():
        p = os.path.join(a.dir, name)
        if not os.path.exists(p):
            out["files"][name] = {"error": "missing"}
            all_ok = False
            continue
        info = inspect(p)
        info["bytes"] = os.path.getsize(p)
        info["rows_expected"] = exp["rows"]
        info["rows_ok"] = info["rows"] == exp["rows"]
        info["cols_ok"] = info["ncols"] == EXPECTED_COLS
        info["bad_rows_ok"] = info["bad_rows"] == 0
        if not a.skip_md5:
            info["md5"] = md5_file(p)
            info["md5_expected"] = exp["md5"]
            info["md5_ok"] = info["md5"] == exp["md5"]
        out["files"][name] = info
        if not (info["rows_ok"] and info["cols_ok"] and info["bad_rows_ok"]):
            all_ok = False

    # 总行数校验：三份之和应等于 45,840,617
    total_rows = sum(v.get("rows", 0) for v in out["files"].values())
    out["total_rows"] = total_rows
    out["total_rows_expected"] = 45840617
    out["total_rows_ok"] = total_rows == 45840617
    if not out["total_rows_ok"]:
        all_ok = False

    out["verdict"] = "PASS" if all_ok else "FAIL"
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
