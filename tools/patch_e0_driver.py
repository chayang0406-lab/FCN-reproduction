#!/usr/bin/env python3
"""对 captain 的 tools/run_e0_baseline.py 做 3 处外科手术式补丁 -> run_e0_baseline_v2.py

为什么用「补丁器」而不是重写：
    重写会让 captain 无法审阅差异。本脚本只做 3 次可见的字符串替换，其余逐字符保留，
    因此 `diff tools/run_e0_baseline.py tools/run_e0_baseline_v2.py` 必须只有 3 个 hunk。
    补丁器自带断言：任一锚点找不到就报错退出，绝不静默漏改。

3 处补丁（每一处都有实测证据，见 docs/运行手册_E0执行与坑_20261118.md）：

  P3  失败的 run 被永久 skip，且统计静默少算
      实测：放一个内容是 traceback 的 train.log ->
            "[skip] e0_exp4_lin1_seed2025 已完成"，summary 里 failed=false、param_check=NO_PARAMS，
            e0_stats.json 里 n_seeds=3 但只有 2 个 run 参与均值。
            => 一次失败会永久污染该种子的结果，且报告里的 n 与实际不符。
      修：只有「valid AUC + test AUC + 参数量正确」才算完成；否则归档旧日志后重跑。
      修：n_seeds 只统计真正有效的 run，并额外给出 n_valid / n_invalid。

  P3b 参数量 MISMATCH 不中止，白跑全部种子
      实测：3 个种子全部 MISMATCH，驱动仍把 3 个 run 全跑完（退出码 0）。
      真实数据 76 min/seed => 8 种子会白烧 10.1 小时，最后才发现配置错。
      修：MISMATCH 立即 sys.exit(2)。

  P4  单轮耗时 / 轮数静默算错（captain 自己也发现过这个坑，但脚本里没改）
      fuxictr fit() 只在开头打一次 "Epoch=1 start"，之后每轮只有 "Epoch=N end"。
      实测：作者 19 轮日志 starts=1 / ends=18 => 原版 n_epochs 恒为 1、mean_epoch_seconds 恒为 None。
      另外 tqdm 进度条写 stderr 且不换行，并流进 train.log 后时间戳不在行首，
      用 `^时间戳` 锚行首会一条都匹配不到（本脚本实测 end_lines=0）。
      修：epoch_1 = end_1 - start_1；epoch_N = end_N - end_{N-1}；正则不锚行首；
          并把 n_epoch_start_lines / n_epoch_end_lines / epoch_seconds 全序列写进 summary，
          让「解析失败返回空列表」能被自证，而不是靠人眼看 mean 是 None。
"""
import os
import re
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_e0_baseline.py")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_e0_baseline_v2.py")

# ---------------------------------------------------------------- P4: 计时解析
OLD_PARSE = '''    n_epoch = len(re.findall(r"Epoch=\\d+ start", txt))
    # 每 epoch 秒数：用 start 时间戳差分
    times = re.findall(r"^(\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d,\\d+) .*Epoch=(\\d+) start",
                       txt, re.M)
    secs = []
    from datetime import datetime
    ts = [(datetime.strptime(t, "%Y-%m-%d %H:%M:%S,%f"), int(e)) for t, e in times]
    ts.sort(key=lambda x: x[1])
    for i in range(1, len(ts)):
        secs.append((ts[i][0] - ts[i - 1][0]).total_seconds())
'''

NEW_PARSE = '''    # [P4] fuxictr fit() **只在第一个 epoch 打 "Epoch=1 start"**，之后每轮只打
    # "Epoch=N end"。用 start 行差分 => secs 恒为空、n_epochs 恒为 1、
    # mean_epoch_seconds 恒为 None，而且**不报错**（静默失效）。
    # 正确算法：epoch_1 = end_1 - start_1；epoch_N = end_N - end_{N-1}。
    #
    # [P4b] 正则**不能**锚行首 `^时间戳`：tqdm 进度条写 stderr 且不换行，并流进
    # train.log 后物理行是
    #   100%|███| 1/1 [00:18<00:00, 18.58s/it]2025-05-13 06:38:23,835 ... Epoch=1 end
    # 时间戳不在行首 => 一条都匹配不到（实测 n_epoch_end_lines=0，依旧静默返回空）。
    _TS = r"(\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d,\\d+)"
    from datetime import datetime
    starts = re.findall(_TS + r" .*?Epoch=(\\d+) start ", txt)
    ends = re.findall(_TS + r" .*?Epoch=(\\d+) end ", txt)
    secs = []
    prev = datetime.strptime(starts[0][0], "%Y-%m-%d %H:%M:%S,%f") if starts else None
    for _ts, _no in ends:
        _cur = datetime.strptime(_ts, "%Y-%m-%d %H:%M:%S,%f")
        if prev is not None:
            secs.append((_cur - prev).total_seconds())
        prev = _cur
    _early = re.search(r"Epoch==?\\d+ early stop", txt)
    # 早停那一轮只打 "early stop"、不打 "end"，所以实际轮数 = end 行数 + 是否早停
    n_epoch = len(ends) + (1 if _early else 0)
    early_m = re.search(r"Epoch==?(\\d+) early stop", txt)

    # [P6] lr 衰减时序。日志的真实顺序是：
    #   Evaluation @epoch 14 - batch 4477:
    #   [Metrics] AUC: 0.812342 - logloss: 0.439460
    #   Monitor(max)=0.812342 STOP!
    #   Reduce learning rate on plateau: 0.000100
    #   ************ Epoch=14 end ************
    # 注意「Epoch=14 end」在这条消息**之后**才打印（因为 checkpoint_and_earlystop
    # 在 train_epoch 内被调用，而 end 行在 fit 的循环体末尾）。
    # 所以归属必须用**紧邻其前的 `Evaluation @epoch N`**，不能用 end 行。
    # （第一版用 end 行，得到 [13,17,18]，全错一位 —— 已修）
    lr_decay_epochs, lr_decay_values = [], []
    for _m in re.finditer(r"Reduce learning rate on plateau: ([\\d.eE+-]+)", txt):
        lr_decay_values.append(float(_m.group(1)))
        _prev = None
        for _em in re.finditer(r"Evaluation @epoch (\\d+) - batch", txt[:_m.start()]):
            _prev = int(_em.group(1))
        lr_decay_epochs.append(_prev)
'''

# ------------------------------------------------- P4c: 自证字段（P4 的一部分）
# 这个坑的本质是「解析失败时返回空列表而不是抛异常」，只看 mean 是 None
# 无法区分「没跑到」和「解析没生效」。把行数与全序列写出来即可自证。
OLD_RETURN = '''        "n_epochs": n_epoch,
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
    }
'''

NEW_RETURN = '''        "n_epochs": n_epoch,
        "mean_epoch_seconds": (sum(secs) / len(secs)) if secs else None,
        "min_epoch_seconds": min(secs) if secs else None,
        "max_epoch_seconds": max(secs) if secs else None,
        # 自证字段：解析没生效时这些会是 0 / []，一眼可见，
        # 不必靠肉眼发现 mean_epoch_seconds 是 None
        "n_epoch_start_lines": len(starts),
        "n_epoch_end_lines": len(ends),
        "epoch_seconds": secs,
        # [P6] lr 衰减时序 —— 「三段式轨迹」的量化指标
        # fuxictr base_model.checkpoint_and_earlystop：
        #   if monitor_value < best + min_delta(=1e-6):   # 本轮没提升
        #       "Monitor(max)=... STOP!" -> if reduce_lr_on_plateau: lr_decay()
        # 实测作者 _016：ep13=0.812382 -> ep14=0.812342（低）-> 衰减 1e-3->1e-4
        #                -> ep15 从 0.812342 跳到 0.815144（+0.0028，占总提升 26%）
        # 因为 min_delta 只有 1e-6，而我们与作者的逐轮 Δ 在 1e-5~1e-4，
        # 所以「哪一轮触发衰减」可能差 1-2 轮，最终 AUC 会因此差 1e-3 量级。
        # 记录这个时序，才能区分「链路不对」与「分岔点漂移」。
        "lr_decay_epochs": lr_decay_epochs,
        "lr_decay_values": lr_decay_values,
        "early_stop_epoch": (int(early_m.group(1)) if early_m else None),
    }
'''

# ------------------------------------------------------- P3: 完整性判据 + 重跑
OLD_SKIP = '''        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            print("[skip] %s 已完成" % run_id, flush=True)
        else:
'''
NEW_SKIP = '''        # [P3] 原版只看 train.log 是否**非空**就 skip。跑崩留下的 traceback 也是
        # 非空，于是失败的种子被永久当成「已完成」，never retried，而 summary 里
        # 只写 failed=false / NO_PARAMS，统计时静默少算一个 n。实测已复现。
        # 改成：必须同时有 valid AUC、test AUC 且参数量正确，才算完成。
        need_run = True
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            if is_run_complete(parse_log(log_path), expect_params):
                print("[skip] %s 已完成（valid/test/参数量三项齐备）" % run_id, flush=True)
                need_run = False
            else:
                _stamp = time.strftime("%Y%m%d-%H%M%S")
                _bad = log_path + ".failed-" + _stamp
                os.replace(log_path, _bad)
                shutil.rmtree(os.path.join(run_dir, "checkpoints"), ignore_errors=True)
                print("[rerun] %s 上次未产出有效结果，旧日志归档为 %s"
                      % (run_id, os.path.basename(_bad)), flush=True)
        if need_run:
'''

# ------------------------------------------- P3: summary 返回字段 + 自检函数
OLD_HELPER_ANCHOR = '''def main():
    ap = argparse.ArgumentParser()
'''

NEW_HELPER = '''def is_run_complete(r, expect_params):
    """[P3] 一个 run 算「已完成」的判据：三项齐备，缺一即重跑。"""
    return (r.get("final_valid_auc") is not None
            and r.get("final_test_auc") is not None
            and r.get("n_params") == expect_params)


def main():
    ap = argparse.ArgumentParser()
'''

# ------------------------------------- P3b: MISMATCH 立即中止 + 统计口径修正
OLD_TAIL = '''        with open(os.path.join(args.workdir, "runs", "e0_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 统计 ----
'''

NEW_TAIL = '''        with open(os.path.join(args.workdir, "runs", "e0_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # [P3b] 参数量不符 / 没跑起来 => 立即中止。原版只打印一行就继续，真实数据
        # 76 min/seed，8 个种子会白烧 10 小时才在最后发现配置没生效。
        if r["param_check"] == "MISMATCH":
            print("\\n!!! 参数量不符（实测 %s，期望 %d），按纪律立即中止，"
                  "剩余种子不再运行 !!!" % (r["n_params"], expect_params), flush=True)
            sys.exit(2)
        if r["param_check"] != "OK":
            print("\\n!!! 本 run 未产出有效结果，已中止（日志：%s）!!!" % log_path, flush=True)
            sys.exit(3)

    # ---- 统计 ----
'''

OLD_STATS = '''    out = {"n_seeds": len(summary),
           "expect_params": expect_params,
           "valid_mean": statistics.mean(va) if va else None,
'''

NEW_STATS = '''    # [P3] n_seeds 必须区分「声明跑了几个」与「真正有几个有效」。
    # 原版把 n_seeds 写成 len(summary)，一旦某 run 无效，报告里 n 会虚报。
    _valid_runs = [r for r in summary if r.get("final_valid_auc") is not None
                   and r.get("final_test_auc") is not None]
    out = {"n_seeds": len(_valid_runs),
           "n_seeds_attempted": len(summary),
           "n_seeds_invalid": len(summary) - len(_valid_runs),
           "invalid_runs": [r["run_id"] for r in summary
                            if r.get("final_valid_auc") is None],
           "expect_params": expect_params,
           "valid_mean": statistics.mean(va) if va else None,
'''


def apply(src, pairs):
    for name, old, new in pairs:
        if src.count(old) != 1:
            sys.exit("!! 锚点 %s 出现 %d 次（应为 1），拒绝改：补丁器与源文件不匹配"
                     % (name, src.count(old)))
        src = src.replace(old, new, 1)
        print("  [applied] %s" % name)
    return src


def main():
    txt = open(SRC, encoding="utf-8").read()
    n_before = len(txt)
    print("源文件 %s (%d 字节)" % (os.path.basename(SRC), n_before))
    out = apply(txt, [
        ("P3  重跑判据（替换 skip 逻辑）", OLD_SKIP, NEW_SKIP),
        ("P3  插入 is_run_complete 辅助函数", OLD_HELPER_ANCHOR, NEW_HELPER),
        ("P3b MISMATCH 立即中止", OLD_TAIL, NEW_TAIL),
        ("P3  统计口径 n_seeds/n_valid", OLD_STATS, NEW_STATS),
        ("P4  单轮耗时解析", OLD_PARSE, NEW_PARSE),
        ("P4c 解析自证字段", OLD_RETURN, NEW_RETURN),
    ])
    open(DST, "w", encoding="utf-8").write(out)
    print("写出 %s (%d 字节, +%d)" % (os.path.basename(DST), len(out), len(out) - n_before))

    # 语法自检
    compile(out, DST, "exec")
    print("语法检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
