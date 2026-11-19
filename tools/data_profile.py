"""数据画像：CTR、字段基数（过滤前）、字段类型分布。

对 Criteo_x4 的 CSV 做流式统计，不整份读入内存。
- --rows N  只统计前 N 行数据行（0 = 全量）
- 输出 JSON，便于写进文档

注意口径：
  这里的"原始基数"是**未过滤**的不同取值个数。
  它不是论文 Table 2 的 vocab_size——后者是 min_categr_count=10 过滤后的值加 2
  （padding_idx + oov_idx）。两者绝不能混进同一列。
"""

import argparse
import json
import sys
from collections import Counter

NUM_I = 13
NUM_C = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    result = {}
    for split in ("train", "valid", "test"):
        path = f"{a.csv_dir}/{split}.csv"
        seen = [set() for _ in range(NUM_I + NUM_C)]
        n = 0
        pos = 0
        empty = [0] * (NUM_I + NUM_C)

        with open(path, "r", encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\r\n").split(",")
            # 期望布局: Label, I1..I13, C1..C26  (40 列)
            for line in f:
                rec = line.rstrip("\r\n").split(",")
                if len(rec) != len(header):
                    continue
                n += 1
                if rec[0] == "1":
                    pos += 1
                for i in range(NUM_I + NUM_C):
                    v = rec[1 + i]
                    if v == "":
                        empty[i] += 1
                    else:
                        seen[i].add(v)
                if a.rows and n >= a.rows:
                    break

        fields = header[1:]
        card = {}
        empty_pct = {}
        for i, name in enumerate(fields):
            card[name] = len(seen[i])
            empty_pct[name] = round(100.0 * empty[i] / n, 2) if n else 0.0

        result[split] = {
            "rows": n,
            "positives": pos,
            "ctr_pct": round(100.0 * pos / n, 4) if n else 0.0,
            "n_fields": len(fields),
            "raw_cardinality": card,
            "empty_pct": empty_pct,
        }
        print(f"{split}: rows={n:,} pos={pos:,} ctr={result[split]['ctr_pct']}%",
              file=sys.stderr)

    # 抽样标注
    result["_meta"] = {
        "rows_sampled": a.rows if a.rows else "all",
        "note": "raw_cardinality 是未过滤基数，不是论文 Table 2 的 vocab_size",
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
