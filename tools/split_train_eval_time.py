"""从作者日志精确拆解：每 epoch 的 train 时间 vs validation eval 时间。

fuxictr 的 fit() 只在循环前打一次 "Epoch=1 start"，循环内只打 "Epoch=N end"，
所以不能靠 start 配对。正确做法：
  epoch N 的起点 = 上一个 "Epoch=N-1 end" 的时间戳（epoch 1 用 "Epoch=1 start"）
  epoch N 的 train 结束 = "Train loss: xxx" 行
  epoch N 的 eval 结束  = 紧随其后的 "[Metrics] AUC: ... logloss: ..." 行
"""
import glob
import os
import re
import statistics
from datetime import datetime


def ts(ln):
    # 【坑】re.match 只匹配行首，与 ^ 锚点是同一个错误；
    # 我们的日志并入 tqdm 后时间戳可能在行中间，必须用 re.search。
    m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)", ln)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def analyze(path):
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    if not lines:
        return None

    # 收集关键时间点
    start1 = None
    ends = []        # (epoch_no, ts)
    trainloss = []   # ts
    metrics = []     # ts
    for ln in lines:
        if "Epoch=1 start" in ln and start1 is None:
            start1 = ts(ln)
        m = re.search(r"Epoch=(\d+) end", ln)
        if m:
            ends.append((int(m.group(1)), ts(ln)))
        if "Train loss:" in ln:
            trainloss.append(ts(ln))
        if "[Metrics]" in ln and "AUC" in ln and "logloss" in ln:
            metrics.append(ts(ln))

    rows = []
    for i, (ep, end_t) in enumerate(ends):
        begin = start1 if i == 0 else ends[i - 1][1]
        if begin is None or end_t is None:
            continue
        # 该 epoch 内最后一个 Train loss 和 Metrics
        tl = [t for t in trainloss if t and begin <= t <= end_t]
        mt = [t for t in metrics if t and begin <= t <= end_t]
        if not tl or not mt:
            continue
        tr = (tl[-1] - begin).total_seconds()
        ev = (mt[-1] - tl[-1]).total_seconds()
        tot = (end_t - begin).total_seconds()
        if tr > 0 and ev >= 0:
            rows.append((ep, tr, ev, tot))
    return rows


def main():
    LOG = os.environ.get("LOG_DIR",
                         r"D:\中南大学\数据挖掘\code\FCN\checkpoints\Criteo_x4_10_h5")
    targets = ["FCNv2_Criteo_016_cde6d2ff.log", "FCNv2_Criteo.log"]
    for name in targets:
        p = os.path.join(LOG, name)
        if not os.path.exists(p):
            continue
        rows = analyze(p)
        print("=" * 78)
        print(name)
        print("=" * 78)
        if not rows:
            print("  (无法解析)")
            continue
        print("%-8s %-12s %-12s %-12s" % ("epoch", "train(s)", "eval(s)", "total(s)"))
        for ep, tr, ev, tot in rows:
            print("%-8d %-12.1f %-12.1f %-12.1f" % (ep, tr, ev, tot))
        trs = [r[1] for r in rows]
        evs = [r[2] for r in rows]
        tots = [r[3] for r in rows]
        # 用最后 5 个 epoch 代表稳态（前几个 epoch 有 JIT/缓存预热）
        n = min(5, len(rows))
        st_tr = statistics.mean(trs[-n:])
        st_ev = statistics.mean(evs[-n:])
        print()
        print("  全部 %d epoch: train 均值 %.1f, eval 均值 %.1f, total 均值 %.1f"
              % (len(rows), statistics.mean(trs), statistics.mean(evs), statistics.mean(tots)))
        print("  末 %d epoch(稳态): train %.1f s, eval %.1f s, 合计 %.1f s"
              % (n, st_tr, st_ev, st_tr + st_ev))
        print("  eval 占 epoch 比例 = %.1f%%" % (100 * statistics.mean(evs) / statistics.mean(tots)))
        print("  eval 最快一次 = %.1f s, 最慢 = %.1f s" % (min(evs), max(evs)))
        print()


if __name__ == "__main__":
    main()

