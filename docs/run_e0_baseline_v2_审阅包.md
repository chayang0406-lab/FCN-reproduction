# `run_e0_baseline_v2.py` 变更审阅包（供 captain 切换前审阅）

> 三份材料：① 5 个 hunk 全文 ② 逐 hunk 副作用说明 ③ 归档逻辑实测证据。
> diff 原文另存：`docs/run_e0_baseline_v2.diff`（6,702 字节）。

---

## 一、先回答两个前置问题

### Q1 `[rerun]` 会不会覆盖已成功的 run？→ **不会，已实测**

判据是 `is_run_complete()`：**valid AUC + test AUC + 参数量正确，三项齐备**才算完成。
只有不满足时才走归档重跑。

**实测方法**：同 seed 连跑两次（Σvocab=910,747 的验证集）。

| | `train.log` md5 | mtime | 目录内容 |
|---|---|---|---|
| 第 1 次 | `c89880b1e3be3541802880c8f05d33ea` | 1795011583 | `checkpoints/ config/ result.json train.log` |
| 第 2 次 | `c89880b1e3be3541802880c8f05d33ea` | 1795011583 | 同上，**无 `.failed-*`** |

第 2 次输出：
```
[skip] e0_exp4_lin1_seed2025 已完成（valid/test/参数量三项齐备）
      valid=0.510487 test=0.495719 params=16532562 (OK)
```
**md5 与 mtime 都没变 → 成功结果绝对安全。** 连 `checkpoints/` 都不会被 `rmtree`。

**反向情形**（无有效结果时才归档）也已实测：
```
[rerun] e0_exp4_lin1_seed2025 上次未产出有效结果，旧日志归档为 train.log.failed-20261118-215020
```
归档名带**秒级时间戳**，多次失败不会互相覆盖。

> 代码原文（hunk3）：
> ```python
> need_run = True
> if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
>     if is_run_complete(parse_log(log_path), expect_params):
>         print("[skip] ... 三项齐备")
>         need_run = False
>     else:
>         _bad = log_path + ".failed-" + _stamp
>         os.replace(log_path, _bad)
>         shutil.rmtree(os.path.join(run_dir, "checkpoints"), ignore_errors=True)
> ```
> 注意 `shutil.rmtree(checkpoints)` **只在 else 分支**，成功路径不触碰。

### Q2 有副作用吗？→ **没有，且已逐项核查**

**超参 / 种子 / 目录结构 / 输出字段，一个都没动**（脚本化核查）：

| 项 | 变更行数 |
|---|---|
| `batch_size` | 0 |
| `learning_rate` | 0 |
| `embedding_regularizer` | 0 |
| `net_dropout` | 0 |
| `num_heads` | 0 |
| `seeds =` | 0 |
| `run_id =` | 0 |
| `model_root` | 0 |
| `early_stop_patience` | 0 |

`epochs` 命中的 1 行是**注释里的 `n_epochs` 字样**，不是超参：
```
+    # "Epoch=N end"。用 start 行差分 => secs 恒为空、n_epochs 恒为 1、
```

---

## 二、5 个 hunk 逐条说明

### hunk1 `@@ -49,16 +49,29 @@` — P4 单轮耗时解析（-9/+22）

**改了什么**：`n_epoch` 与 `secs` 的计算方式。
- 原：用 `Epoch=N start` 行差分 → fuxictr 只打一次 start → `secs` 恒空
- 新：`epoch_1 = end_1 - start_1`；`epoch_N = end_N - end_{N-1}`
- 正则从 `^时间戳`（锚行首）改为**不锚行首**（tqdm 进度条并流后时间戳不在行首）
- `n_epoch = len(ends) + (1 if 早停 else 0)`（早停轮只打 `early stop`，不打 `end`）

**副作用评估**：只影响 `n_epochs` / `mean_epoch_seconds` 两个**输出字段**，
不触碰训练、不触碰 `param_check`、不触碰 valid/test 解析。

### hunk2 `@@ -67,9 +80,23 @@` — 解析自证字段（-0/+14）

纯**新增** 5 个返回字段，无删除：
```python
"min_epoch_seconds" / "max_epoch_seconds"
"n_epoch_start_lines" / "n_epoch_end_lines" / "epoch_seconds"
```
**理由**：这个坑的本质是「解析失败返回空列表而不是抛异常」。把行数写进产出，
`n_epoch_end_lines=0` 能自证解析失效，不必靠人眼看 mean 是 None。

**副作用评估**：`e0_summary.json` 会多 5 个键。若有下游脚本按「键集合严格相等」校验
会受影响——但 `e0_guard.py` 是按名取键，实测兼容。

### hunk3 `@@ -102,9 +129,23 @@` — P3 重跑判据（-2/+16）

**改了什么**：`if 日志非空: skip` → `if 日志三项齐备: skip else 归档重跑`
新增 `is_run_complete()` 辅助函数（hunk2 与 hunk3 之间）。

**副作用评估**：见 Q1。成功路径行为完全不变（实测 md5/mtime 未变）。

### hunk4 `@@ -207,11 +248,29 @@` — P3b 中止 + 统计口径（-1/+19）

**改了什么**：
1. `param_check == "MISMATCH"` → `sys.exit(2)`（原来只打印一行继续）
2. `param_check != "OK"` → `sys.exit(3)`
3. `n_seeds` 从 `len(summary)` 改为「有效 run 数」，并新增 3 个字段

**副作用评估**：这是**行为变更中最重要的一个**——需要注意：
- **`sys.exit(2)` 会让 driver 提前退出**。如果你的启动脚本包了 `while` 重试循环，
  会被这个退出码打断。建议：把 exit=2 视为「配置错，需人工修」，**不要自动重试**。
- 正常跑通时（8 个种子全 OK）这个分支永不触发，退出码仍是 0。

新增 3 个 JSON 字段：
```python
"n_seeds"           # 有效 run 数（原为 len(summary)）
"n_seeds_attempted" # 实际尝试数
"n_seeds_invalid"   # 无效数
"invalid_runs"      # 无效 run_id 列表
```

### hunk5 `@@ -240,4 +299,4 @@` — 文件末尾换行（-1/+1）

原文件末尾是 CRLF（`^M$`），新文件是 LF（`$`）。**无语义差异**，
是上游文件以 CRLF 保存、我写入时统一为 LF 的结果。

如你希望保留 CRLF，我可以改；但我建议保留 LF（Linux 下更规范，
且 `tr -d '\r'` 的管线里 CRLF 是历史坑源）。

---

## 三、切换前后的完整验证清单（都已实测通过）

| 验证项 | 结果 |
|---|---|
| 语法 | `compile()` 通过 |
| 闸门**放行** | Σvocab=910,747 → `params=16532562 (OK)`，2 种子 |
| 闸门**拦截** | Σvocab=2,359 → `MISMATCH` → `exit=2`，只跑 1 个 run |
| 崩溃 run 重跑 | `[rerun]` + 归档 + 重跑成功 |
| 成功 run 不覆盖 | md5/mtime 未变，无 `.failed-*` |
| 三份作者日志 | 20/19/21 轮，191.22 / 204.38 / 219.73 s |
| 指标不变 | valid=0.510487 / test=0.495719 与原版一致 |
| 超参未动 | 9 项逐项核查，变更行数全 0 |

---

## 四、切换方式

```bash
P=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
# 原文件保留不动，只换调用名
nohup /usr/local/bin/python3 -X utf8 $P/tools/run_e0_baseline_v2.py \
  --repo $P/code/FCN_repo --workdir $P/runs --data-root $P/data \
  --seeds 2025,2026,2027,2028,2029,2030,2031,2032 \
  --exp-layers 4 --lin-layers 1 > $P/runs/e0_driver.log 2>&1 &
```

**若你想自己重新生成**（而不信任我的二进制文件）：
```bash
/usr/local/bin/python3 -X utf8 $P/tools/patch_e0_driver.py
# 输出：6 个 [applied]，写出 run_e0_baseline_v2.py，语法检查通过
# 补丁器自带断言：任一锚点出现次数 ≠ 1 就拒绝改，绝不静默漏改
```
