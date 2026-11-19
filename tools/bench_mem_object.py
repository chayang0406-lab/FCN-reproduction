"""关键测量：fuxictr 的 read_csv 用 dtype=object 读，内存远大于默认推断。

fuxictr/preprocess/feature_processor.py:69
    ddf = pd.read_csv(data_path, sep=sep, usecols=usecols_fn,
                      dtype=object, memory_map=True, nrows=nrows)

dtype=object 意味着**每一格都是 Python 对象指针 + 独立 str 对象**，
26 个字符串列在高基数下会产生海量小对象。这与我先前用默认 dtype 推断
测出的 1.66 KB/row 完全不同。必须按真实路径重测。

同时测「preprocess 之后」「transform 之后」两个中间态，还原峰值内存。
"""
import os
import resource
import sys

import numpy as np
import pandas as pd

SUB = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/data/_subset_for_dcu/train.csv"
N = 200000
REAL_TRAIN = 36672493
MEM_LIMIT_GIB = 114.97


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


print("=" * 84)
print("按 fuxictr 真实路径测内存（dtype=object）")
print("=" * 84)

base = rss_gib()
# 完全照抄 fuxictr 的 read_csv 参数
ddf = pd.read_csv(SUB, sep=",", dtype=object, memory_map=True, nrows=N)
after_read = rss_gib()
print("  读入 %d 行 (dtype=object):" % N)
print("    进程 RSS  %.3f GiB (基线 %.3f)" % (after_read, base))
print("    ddf deep  %.3f GiB  => %.2f KB/row"
      % (ddf.memory_usage(deep=True).sum() / 1024 ** 3,
         ddf.memory_usage(deep=True).sum() / N / 1024))

per_row_kb = ddf.memory_usage(deep=True).sum() / N / 1024
est = per_row_kb * REAL_TRAIN / 1024 ** 2   # GiB
print()
print("  外推 train 36,672,493 行:")
print("    deep 内存 = %.1f GiB   (上限 %.1f GiB, 占 %.0f%%)"
      % (est, MEM_LIMIT_GIB, 100 * est / MEM_LIMIT_GIB))

# 对比默认 dtype
del ddf
import gc
gc.collect()
ddf2 = pd.read_csv(SUB, nrows=N)
default_kb = ddf2.memory_usage(deep=True).sum() / N / 1024
print()
print("  对照：默认 dtype 推断 = %.2f KB/row（外推 %.1f GiB）"
      % (default_kb, default_kb * REAL_TRAIN / 1024 ** 2))
print("  => dtype=object 比默认推断大 %.2f 倍" % (per_row_kb / default_kb))

print()
print("=" * 84)
print("结论")
print("=" * 84)
if est > MEM_LIMIT_GIB * 0.8:
    print("  !! dtype=object 读入 train 单份就需要 %.1f GiB，超过上限 80%%。" % est)
    print("     而且 build_dataset 在 preprocess 时还会再产生一份副本（ddf.loc 切片），")
    print("     峰值可能达到 2 倍 -> %.1f GiB，**必然 OOM**。" % (est * 2))
    print("     => 必须分块（data_block_size > 0），或用 nrows 分批转。")
else:
    doubled = est * 2
    verdict = "仍安全" if doubled < MEM_LIMIT_GIB * 0.85 else "接近/超过上限，建议分块"
    print("  dtype=object 单份 %.1f GiB，尚在安全范围；但 preprocess 会再复制一份，" % est)
    print("  峰值按 2 倍估 %.1f GiB -> %s" % (doubled, verdict))

