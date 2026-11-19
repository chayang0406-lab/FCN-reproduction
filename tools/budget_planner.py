"""全量 Criteo 数据处理的磁盘/内存/时间预算测算。

所有输入都来自实测，不猜：
  - h5 每行字节   : 320.0 B/row（4 个规模实测，200k~2000k 完全一致，非常稳）
  - CSV 每行字节  : 242.3 B/row（20 万行子集实测）
  - 模型文件大小  : 63.1 MB（16,532,562 参数 × 4B，DCU 上实际 torch.save 实测）
  - 转换速度      : 见 csv2h5_scaling.json（200k~2000k 四点拟合）
  - 转换内存      : RSS = 118 MB + 1.2 KB/row（实测拟合）
  - 磁盘余量      : 实测 df

输出：时间/磁盘/内存三张预算表 + 决策建议。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/bench"

# ---- 实测常量 ----
H5_BYTES_PER_ROW = 320.0
CSV_BYTES_PER_ROW = 242.3
MODEL_MB = 63.1
ZIP_MB = 4049          # 4,049,391,068 B = 3861 MiB = 4.05 GB（十进制）

# 数据规模（前置事实已核实）
N_TRAIN = 36672493
N_VALID = 4584062
N_TEST = 4584062
N_ALL = N_TRAIN + N_VALID + N_TEST      # 45,840,617

# 内存上限（cgroup v1 实测）
MEM_LIMIT_GIB = 123480309760 / 1024 ** 3    # 114.97 GiB

# 磁盘：实测 df
DISK_AVAIL_GB = 163.0    # 采集时 ks_p300s_public 可用
PROJECT_NOW_GB = 4.7     # 当前项目已占


def gb(nbytes):
    return nbytes / 1024 ** 3


def gib(nbytes):
    return nbytes / 1024 ** 3


def load_scaling():
    p = os.path.join(BENCH, "csv2h5", "csv2h5_scaling.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


def main():
    print("=" * 88)
    print("全量 Criteo 数据处理预算测算（所有常量均为实测）")
    print("=" * 88)
    print("  数据规模: train=%d valid=%d test=%d 合计=%d" % (N_TRAIN, N_VALID, N_TEST, N_ALL))
    print("  h5 每行    : %.1f B（4 个规模实测一致）" % H5_BYTES_PER_ROW)
    print("  CSV 每行   : %.1f B（实测）" % CSV_BYTES_PER_ROW)
    print("  模型文件   : %.1f MB（实测 torch.save）" % MODEL_MB)
    print("  内存上限   : %.1f GiB（cgroup 实测）" % MEM_LIMIT_GIB)
    print("  磁盘可用   : %.0f GB（实测 df）" % DISK_AVAIL_GB)

    # ---------- 1. 磁盘 ----------
    print()
    print("=" * 88)
    print("【1】磁盘预算")
    print("=" * 88)
    csv_bytes = CSV_BYTES_PER_ROW * N_ALL
    h5_bytes = H5_BYTES_PER_ROW * N_ALL

    rows = [
        ("zip 压缩包", ZIP_MB / 1024),
        ("解压后 3 个 CSV", gb(csv_bytes)),
        ("转换后 3 个 h5", gb(h5_bytes)),
    ]
    tot_stage = sum(v for _, v in rows)
    print("  %-24s %10s" % ("项目", "大小"))
    print("  " + "-" * 40)
    for name, v in rows:
        print("  %-24s %10.2f GB   %s" % (name, v, ""))
    print("  " + "-" * 40)
    print("  %-24s %10.2f GB   <- 若三者同时存在" % ("小计（并存）", tot_stage))

    # 模型
    for n_models, label in ((30, "7种子+E3/E4 约30个"), (21, "仅7种子+E0"), (7, "仅7种子")):
        mv = n_models * MODEL_MB / 1024
        print("  %-24s %10.2f GB   %s" % ("checkpoints", mv, label))

    print()
    print("  --- 关键结论 ---")
    avail = DISK_AVAIL_GB - PROJECT_NOW_GB
    print("  当前可用（扣除项目已占 %.1f GB）= %.1f GB" % (PROJECT_NOW_GB, avail))
    print()
    print("  方案 A（最保守：zip+CSV+h5 全留）:")
    needA = tot_stage + 30 * MODEL_MB / 1024
    print("    需要 %.1f GB，剩余 %.1f GB -> %s" % (
        needA, avail - needA, "够" if needA <= avail else "不够！"))
    print()
    print("  方案 B（推荐：转完 h5 立即删 CSV）:")
    needB = ZIP_MB / 1024 + gb(h5_bytes) + 30 * MODEL_MB / 1024
    print("    需要 %.1f GB（zip + h5 + 30模型），剩余 %.1f GB -> %s" % (
        needB, avail - needB, "够" if needB <= avail else "不够！"))
    print()
    print("  方案 C（最省：转完 h5 后 zip 与 CSV 都删）:")
    needC = gb(h5_bytes) + 30 * MODEL_MB / 1024
    print("    需要 %.1f GB，剩余 %.1f GB -> %s" % (
        needC, avail - needC, "够" if needC <= avail else "不够！"))

    # ---------- 2. 内存 ----------
    print()
    print("=" * 88)
    print("【2】内存预算：pandas 一次性读入会不会爆？")
    print("=" * 88)
    sc = load_scaling()
    if sc:
        a_mb = sc["fit_rss_a_mb"]
        b_mb_per_row = sc["fit_rss_kb_per_row"]   # 注意：json 里字段名虽叫 kb，实测单位是 MB/row
        print("  实测拟合: RSS(n) = %.0f MB + %.4f MB/row (= %.2f KB/row) x n" % (a_mb, b_mb_per_row, b_mb_per_row*1024))
        for label, n in (("train 单份", N_TRAIN), ("全量三份(逐份转)", N_ALL)):
            rss_gib = (a_mb + b_mb_per_row * n) / 1024
            print("    %-16s %d 行 -> RSS %.1f GiB  (上限 %.1f GiB)  %s"
                  % (label, n, rss_gib, MEM_LIMIT_GIB,
                     "安全" if rss_gib < MEM_LIMIT_GIB * 0.8 else "接近上限，需分块"))
        print()
        print("  说明：build_dataset 是逐份处理（train/valid/test 分开读），")
        print("        不会三份同时载入内存。所以关键看单份最大的 train。")
        rss_train = (a_mb + b_mb_per_row * N_TRAIN) / 1024
        print("        train 单份预期 RSS = %.1f GiB，占上限 %.0f%%。"
              % (rss_train, 100 * rss_train / MEM_LIMIT_GIB))
        if rss_train < MEM_LIMIT_GIB * 0.8:
            print("        -> 不需要分块，一次性读入可行。")
        else:
            print("        -> 建议分块。")
    else:
        print("  (缺 csv2h5_scaling.json，跳过)")

    # ---------- 3. 时间 ----------
    print()
    print("=" * 88)
    print("【3】CSV -> h5 时间预算")
    print("=" * 88)
    if sc:
        a = sc["fit_seconds_a"]
        b_us = sc["fit_seconds_per_row_us"]
        b = b_us / 1e6
        print("  实测四点（%s）:" % ", ".join(str(p["n_rows"]) for p in sc["points"]))
        for p in sc["points"]:
            print("    n=%-9d %8.2f s  => %.1f us/row" % (
                p["n_rows"], p["seconds"], p["seconds"] / p["n_rows"] * 1e6))
        print()
        print("  拟合: t(n) = %.1f s + %.1f us/row × n" % (a, b_us))

        # 乐观/悲观：用实测点里的最小/最大 us/row 作为区间
        uss = [p["seconds"] / p["n_rows"] * 1e6 for p in sc["points"]]
        us_best, us_worst = min(uss), max(uss)
        print()
        print("  --- 乐观/悲观区间（用实测点里的最快/最慢 per-row 速率）---")
        for label, n in (("train 单份", N_TRAIN), ("三份合计", N_ALL)):
            t_opt = (a + (us_best / 1e6) * n) / 60
            t_fit = (a + b * n) / 60
            t_pes = (a + (us_worst / 1e6) * n) / 60
            print("    %-12s 乐观 %.0f min | 拟合 %.0f min | 悲观 %.0f min  (%.1f~%.1f h)"
                  % (label, t_opt, t_fit, t_pes, t_opt / 60, t_pes / 60))

        print()
        print("  --- 不确定来源（必须写进文档）---")
        print("    1) 内存压力：200 万行 RSS 仅 3 GB，3670 万行 ~49 GiB，")
        print("       pandas 大对象分配、GC、swap 行为会变，per-row 成本可能上升。")
        print("    2) I/O：bench 的 CSV 在同一磁盘且已被 page cache 缓存；")
        print("       真实转换时 CSV 是冷读，且同时要写 ~14 GB h5，I/O 争用更大。")
        print("    3) min_categr_count 过滤：真实数据有大量长尾类别，")
        print("       编码器要做 dict 查找与重映射，比合成重复行更慢。")
        print("    4) 真实数据的字符串唯一值远多于复制的子集（基数高 -> dict 更大 -> 更慢）。")
        print("    => 真实耗时很可能落在【悲观】一侧。建议按悲观值排期。")

    # ---------- 4. 建议 ----------
    print()
    print("=" * 88)
    print("【4】决策建议")
    print("=" * 88)
    print("  磁盘: 采用方案 B/C —— 转完 h5 立即删 CSV（省 %.1f GB），zip 保留到 h5 校验通过再删。" % gb(csv_bytes))
    print("  内存: 单份处理，不需要分块；但转换期间不要并行跑训练（会叠加 RSS）。")
    print("  时间: 串行转三份，按悲观值预留；建议 nohup 后台跑并留日志。")
    print("  顺序: 先转 valid/test（小，快，可早发现问题），再转 train（大，慢）。")
    print("        这样若配置有错，15 分钟内就能发现，而不是等 2 小时。")


if __name__ == "__main__":
    main()

