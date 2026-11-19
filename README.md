# FCN 复现与大数理处理实验（海光 DCU 单卡）

复现 **FCN: Fusing Exponential and Linear Cross Network for CTR Prediction**（KDD'26，DOI `10.1145/3770854.3780177`，arXiv `2407.13349v8`），
并在海光 DCU K500SM_AI 单卡上完成多种子噪声基线与两处小改进。

> **本文档是项目索引，不是课程报告。** 课程设计报告见 [`report/课程设计报告.md`](report/课程设计报告.md)。

---

## 一、当前状态（截至 2026-11-19）

| 项 | 状态 |
|---|---|
| 数据管道（CSV → h5） | ✅ 完成，Σvocab = 910,747 与作者日志 39/39 字段逐一相符 |
| E0 基线（单种子） | ✅ 完成，valid AUC 0.815598 / test 0.816158 |
| E0 基线（8 种子） | 🔄 运行中（seed2025 复用，实跑 2026–2032） |
| E4-B LCA 压缩率扫描 | ⏳ 未执行 |
| E3 Tri-BCE 配对检验 | ⏳ 未执行 |
| E4-A 组件消融 | ⏳ 未执行 |

**2026-11-19 02:00–14:45 曾发生一次节点故障**（SSH 在密钥交换阶段失败，96 分钟），
实验中断；恢复后数据与已完成结果完好（共享盘持久化），已重跑。
经过见 [`docs/节点故障与恢复指引_20261119.md`](docs/节点故障与恢复指引_20261119.md)。

---

## 二、目录结构

```
course_ctr_fcn_20261118/
├── code/
│   ├── FCN_repo/           # 作者原始仓库（零改动，作为对照基准）
│   ├── src/                # 改进实现（LCA 压缩率参数化 + Tri-BCE 软权重）
│   └── configs/            # 15 个 expid 配置
├── tools/                  # 工具脚本（35 个），见下方说明
├── scripts/                # 早期探索脚本
├── docs/                   # 39 份过程文档（实测记录、坑位、方法论）
├── report/                 # 课程设计报告
├── configs/
├── data/                   # 不入库：Criteo_x4 zip/CSV/h5，共 28 GB
└── runs/                   # 不入库：实验输出
```

### `tools/` 重点脚本

| 脚本 | 用途 |
|---|---|
| `build_h5.py` | CSV → h5，含双硬断言（Σvocab、参数量） |
| `run_e0_baseline_v2.py` | E0 driver（修复官方 3 处缺陷） |
| `e0_startup_check.py` | 启动 5 项硬断言 + 解析交叉验证 |
| `author_baseline.py` | 作者基准常量（与日志交叉核验） |
| `verify_h5_dataset.py` | h5 独立验收（第三方，不采信转换器结论） |
| `verify_improved_configs.py` | 15 配置 4 维闸门（参数量闸门对 8/15 配置是盲的） |
| `paired_analysis.py` | 配对检验（含配对性硬校验） |
| `run_improved_driver.py` | E3/E4 driver |
| `param_formula.py` | 参数量闭式公式 |

---

## 三、环境

```
加速卡 : 海光 DCU K500SM_AI (gfx928, 120 CU, 63.98 GiB)
DTK    : 26.04      需先 source /opt/dtk-26.04/env.sh
PyTorch: 2.7.1+das.opt1.dtk2604
Python : 3.11.9
框架   : fuxictr==2.0.1
关键锁 : scikit-learn==1.4.2（≥1.5 移除了 log_loss 的 eps 参数，会在第 2 轮之后崩）
         numpy==1.25.0
```

**该节点无外网出口**，所有 wheel 需本地下载后上传安装（`pip install --no-index --find-links`）。

---

## 四、数据重建

```bash
# 1. 下载 Criteo_x4（4.05 GB）
#    https://huggingface.co/datasets/reczoo/Criteo_x4  →  Criteo_x4.zip
#    md5 应为 41ca8e3ef1fa1dadf95132f5e11570e1

# 2. 解压到 data/csv/（应得 train 36,672,493 / valid 4,584,062 / test 4,584,062 行）

# 3. 转 h5（约 64 分钟，峰值内存 45 GB）
python3 tools/build_h5.py --csv-dir data/csv --out data/Criteo_x4_10_h5 \
        --data-block-size 0

# 4. 独立验收
python3 tools/verify_h5_dataset.py --data-dir data/Criteo_x4_10_h5 --strict
```

> **重跑前必须先 `rm -f data/Criteo_x4_10_h5/feature_map.json`**。
> 残留的 feature_map.json 会让 `build_dataset` 打 `Skip rebuilding` 并静默跳过整个转换
> ——**「秒完成并报成功」但 h5 是空的**，比直接报错更危险。

---

## 五、跑实验

```bash
# E0 基线（8 种子）
python3 tools/run_e0_baseline_v2.py \
  --repo code/FCN_repo --workdir . --data-root data \
  --seeds 2025,2026,2027,2028,2029,2030,2031,2032 \
  --exp-layers 4 --lin-layers 1 --epochs 30
```

`--epochs 30` 是**安全上限**（作者的 19 轮是早停结果，不是最优轮数）。
实测首种子 23 轮自然早停，未触及上限。**上限是否被触及须如实报告。**

---

## 六、关键事实（勿重复调研）

| 事实 | 值 |
|---|---|
| 复现基准日志 | `code/FCN_repo/checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_016_cde6d2ff.log` |
| 基准配置 | exp=4 / lin=1，16,532,562 参数 |
| 基准精度 | valid AUC 0.815627 / test AUC 0.816145 |
| 基准耗时 | 19 轮早停，204.38 s/epoch（RTX 4090） |

**注意：作者仓库里有两批日志，来自两个不同的代码库。**
同目录的 `FCNv2_Criteo.log` 是旧实现（3ECN+3LCN，16,138,194 参数），
**用 HEAD 代码在任何配置下都复现不出这个参数量**，不能作基准。
辨析见 [`docs/`](docs/) 下相关文档与报告附录 G。

---

## 七、方法论沉淀

本项目最独特的部分不是实验数字，而是一套**自查机制**：

- [`docs/方法论教训_20261119.md`](docs/方法论教训_20261119.md) —— 17 类错误，按「输出看着对但结果不对 / 数字对但对象错 / 判断阈值错」三类分组，每条含根因与防护措施
- 报告附录 I —— 数值核对方法学：一次真实的手工填表错误及其后果量化

核心原则：

> **数字对了不等于结论对了；数字错了也不等于结论错了。**
> 后者更危险——拿错的计数去支撑对的结论，它会显得更可信，因为没人会去核一个"看起来精确"的数字。

---

## 八、引用

```bibtex
@inproceedings{fcn2026,
  title     = {FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770854.3780177}
}
```
