"""在真实 CSV 上预测 I 字段的 bucketize 词表，早于全量 h5 转换完成就能发现问题。

为什么需要这个脚本：
全量 CSV->h5 要跑 2-3 小时。如果 bucketize 规则用错（例如按论文正文用 log2），
要等 3 小时后才在 vocab 校验里发现，浪费整个窗口。
这个脚本直接对 CSV 抽样应用 fuxictr 的 bucketize 规则，**几分钟就能判断规则对不对**。

判定依据：
- 用 ⌊(ln x)²⌋（作者实现）: I1 在 1M 抽样下应接近 44，全量应为 44
- 用 ⌊log2 x⌋（论文正文）  : I1 上界只有 19，抽样下约 12-16

用法:
    python3 predict_bucket_vocab.py --csv <train.csv> --rows 1000000
"""
import argparse
import json
import math
import sys


def bucket_code(x):
    """fuxictr/datasets/criteo.py 的原始规则，逐字复刻。"""
    if x > 2:
        return int(math.floor(math.log(x) ** 2))
    return int(x)


def bucket_log2(x):
    """论文正文写的规则，仅用于对比。"""
    if x > 0:
        return int(math.floor(math.log2(x)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rows", type=int, default=1000000)
    ap.add_argument("--min-count", type=int, default=10)
    a = ap.parse_args()

    # I1..I13 在 CSV 中的列索引（0=Label, 1..13 = I1..I13）
    n_i = 13
    counts_code = [{} for _ in range(n_i)]
    counts_log2 = [{} for _ in range(n_i)]
    empty = [0] * n_i
    n = 0

    with open(a.csv, "r", encoding="utf-8", newline="") as f:
        f.readline()  # header
        for line in f:
            rec = line.rstrip("\r\n").split(",")
            if len(rec) < 14:
                continue
            n += 1
            for i in range(n_i):
                v = rec[1 + i]
                if v == "":
                    empty[i] += 1
                    x = 0.0    # fill_na: 0
                else:
                    x = float(v)
                c = bucket_code(x)
                counts_code[i][c] = counts_code[i].get(c, 0) + 1
                l = bucket_log2(x)
                counts_log2[i][l] = counts_log2[i].get(l, 0) + 1
            if a.rows and n >= a.rows:
                break

    print(f"sampled rows = {n:,}  (fill_na=0 applied to empty values)")
    print()
    hdr = f"{'field':<6}{'code:raw':>10}{'code:>=k':>10}{'log2:raw':>10}{'log2:>=k':>10}{'empty%':>9}"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for i in range(n_i):
        rc = len(counts_code[i])
        rl = len(counts_log2[i])
        fc = sum(1 for v in counts_code[i].values() if v >= a.min_count)
        fl = sum(1 for v in counts_log2[i].values() if v >= a.min_count)
        name = f"I{i+1}"
        ep = 100.0 * empty[i] / n if n else 0
        print(f"{name:<6}{rc:>10}{fc:>10}{rl:>10}{fl:>10}{ep:>8.2f}%")
        out[name] = {"code_raw": rc, "code_filtered": fc,
                     "log2_raw": rl, "log2_filtered": fl,
                     "empty_pct": round(ep, 2)}

    print()
    print("判读方法：")
    print("  作者全量 vocab: I1=44 I2=99 I3=122 I4=41 I5=220 I6=112 I7=80")
    print("                  I8=69 I9=92 I10=6 I11=27 I12=37 I13=72")
    print("  抽样值必然 <= 全量值（抽样见不到稀有的桶）。")
    print("  若 code 列明显接近作者值、log2 列远小 -> 规则正确。")
    print("  若 code 列也只有 log2 量级（I1 约 12-16） -> bucketize 用错了，立刻停。")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
