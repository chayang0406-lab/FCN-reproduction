#!/usr/bin/env python3
"""实测 fuxictr CSV 读取的**峰值**内存，判断 4584 万行会不会 OOM。

## 为什么第一版设计错了（保留教训）
初版在**同一进程内**循环测多档行数，用「读完之后 RSS - 读之前 RSS」当内存增量。
结果 R²=0.0009（几乎零相关），且 200k 行的增量比 100k 行还小。
根因：pandas/glibc 释放的内存不还给 OS，下一轮复用了已分配的 arena，
所以「差值」既不是峰值也不是真实占用。**这是典型的测量方法错误，会得出错误结论。**

## 正确做法
每一档行数都**起一个全新子进程**，用 `resource.getrusage(RUSAGE_SELF).ru_maxrss`
读该进程的**历史峰值 RSS**（这才是 OOM 杀手看的东西），进程退出后记录。
这样每档都是干净测量，不受前面档位污染。

用法：
  # 逐档单独跑（推荐，由 shell 循环调用）
  python3 probe_csv_memory.py --train-csv X --nrows 1000000
  # 或一次性跑完所有档（内部起子进程）
  python3 probe_csv_memory.py --train-csv X --sweep 500000,1000000,2000000
"""
import argparse
import json
import os
import resource
import subprocess
import sys


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_once(csv_path, nrows, dtype_mode):
    """在当前进程里读一次，返回 (峰值RSS_MB, 实际行数, 列数)。"""
    import pandas as pd
    before = peak_mb()
    if dtype_mode == "object":
        df = pd.read_csv(csv_path, dtype=object, memory_map=True,
                         nrows=nrows, na_filter=False)
    else:
        df = pd.read_csv(csv_path, memory_map=True, nrows=nrows)
    peak = peak_mb()
    info = (peak, len(df), len(df.columns))
    # 额外测：preprocess 后 astype(int) 的峰值
    del df
    return info


def child_main(args):
    peak, rows, cols = measure_once(args.train_csv, args.nrows, args.dtype_mode)
    out = {"nrows": rows, "cols": cols, "peak_rss_mb": peak,
           "dtype_mode": args.dtype_mode, "pid": os.getpid()}
    print(json.dumps(out))


def sweep(args):
    """每一档起独立子进程，保证测量干净。"""
    ns = [int(x) for x in args.sweep.split(",")]
    results = []
    here = os.path.abspath(__file__)
    for n in ns:
        cmd = [sys.executable, "-X", "utf8", here,
               "--train-csv", args.train_csv, "--nrows", str(n),
               "--dtype-mode", args.dtype_mode, "--child"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
        if not line:
            print("  档 %d 失败: %s" % (n, p.stderr[-300:]), flush=True)
            continue
        r = json.loads(line[-1])
        results.append(r)
        print("  nrows=%-9d peak=%.1f MB  (%.2f 字节/行)"
              % (r["nrows"], r["peak_rss_mb"],
                 r["peak_rss_mb"] * 1024 * 1024 / max(r["nrows"], 1)), flush=True)

    if len(results) < 2:
        print("\n有效档位不足，无法拟合")
        return

    import numpy as np
    xs = np.array([r["nrows"] for r in results], dtype=float)
    ys = np.array([r["peak_rss_mb"] for r in results], dtype=float)
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print()
    print("线性拟合: peak(MB) = %.8f × rows + %.1f" % (slope, intercept))
    print("R² = %.6f  %s" % (r2, "(线性性成立，外推可信)" if r2 > 0.98
                             else "(⚠️ 线性性不足，外推不可信！)"))
    print("每行开销 = %.2f 字节" % (slope * 1024 * 1024))

    est = (slope * args.full_rows + intercept) / 1024.0
    print()
    print("外推全量 %d 行 ≈ %.1f GB（占上限 %.1f%%）"
          % (args.full_rows, est, est / args.mem_limit_gb * 100))
    verdict = "SAFE" if est < args.mem_limit_gb * 0.6 else \
              ("RISK" if est < args.mem_limit_gb else "OOM")
    print("判定: %s" % verdict)
    print("\nJSON_RESULT " + json.dumps({
        "points": results, "slope_mb_per_row": slope, "intercept_mb": intercept,
        "r2": r2, "est_full_gb": est, "mem_limit_gb": args.mem_limit_gb,
        "verdict": verdict, "dtype_mode": args.dtype_mode,
        "full_rows": args.full_rows}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--nrows", type=int, default=0)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--dtype-mode", default="object", choices=["object", "infer"])
    ap.add_argument("--full-rows", type=int, default=36672493)
    ap.add_argument("--mem-limit-gb", type=float, default=114.97)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()

    if args.child:
        child_main(args)
    elif args.sweep:
        print("每档独立子进程测量峰值 RSS（dtype=%s）" % args.dtype_mode)
        sweep(args)
    else:
        child_main(args)


if __name__ == "__main__":
    main()

