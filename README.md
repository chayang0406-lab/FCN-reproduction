# FCN 复现与环境迭代（海光 DCU / NVIDIA RTX 4090 双平台）

复现 **FCN: Fusing Exponential and Linear Cross Network for CTR Prediction**（KDD'26，DOI `10.1145/3770854.3780177`，arXiv `2407.13349v8`），
在现有开源实现基础上做**可移植性/性能优化**，并补齐论文要求的多数据集与基线对比。

> 本文档是项目索引。课程报告见 [`report/课程设计报告.md`](report/课程设计报告.md)。

**仓库**：https://github.com/buzhimingLF/FCN-Fusing-Exponential-and-Linear-Cross-Network-for-Click-Through-Rate-Prediction

---

## 一、当前状态（2026-11-21）

### 1.1 复现结果（iPinYou / KKBox 已完成）

在 **NVIDIA RTX 4090 单卡**上，用作者公开代码与超参，两个数据集均复现成功：

| 数据集 | 模型 | TEST AUC | TEST Logloss | 论文报告 | 作者本人日志 | 判定 |
|---|---|---|---|---|---|---|
| **iPinYou** | FCNv1 (=FCNp) | **0.7848257** | **0.0055336** | 0.7848 / 0.005529 | 0.784827 / 0.005534 | ✅ 差 1e-6 |
| **KKBox** | FCNv1 (=FCNp) | **0.8566949** | **0.4747218** | 0.8567 / 0.4754 | 0.855871 / 0.475116 | ✅ 差 0.000005 |
| **KKBox** | FCNv2 (=FCNsp) | **0.8551047** | 0.4811236 | 0.8551 / 0.4754 | — | ✅ AUC 精确吻合 |

**iPinYou 的强证据**：参数量 21,537,982 与作者日志**完全相同**，早停轮次同为第 19 轮，逐 epoch 曲线最大偏差 1e-4。

> 完整报告：[`report/复现报告.md`](report/复现报告.md)

### 1.2 改进探索：D-FCN（分支去相关正则）

复现完成后，针对"融合层增益被分支同质化卡住"的问题做了一次改进尝试，结论是**条件有效**：

| 数据集 | 基线 AUC | D-FCN AUC | 分支相关（基线→改进） | 判定 |
|---|---|---|---|---|
| iPinYou | 0.784826 | **0.785622**（λ=0.05） | 0.931 → −0.020 | ✅ +0.0008 |
| KKBox | **0.856695** | 0.854224（λ=0.05） | 0.967 → 0.369 | ❌ −0.0025 |

**KKBox 的 λ 扫描（0 / 0.002 / 0.005 / 0.05）**：AUC 分别为 0.856695 / 0.853945 / 0.854192 / 0.854224——一旦引入正则即损失约 0.0025，且加大 λ 不会更糟也不回升。

**适用边界**：去相关正则只在"分支贡献失衡"时有益（iPinYou 的 LCN 权重激活率 99%，几乎不贡献）；当两分支已均衡贡献时（KKBox 约 50/50），任何 λ 都有害，最优取 **λ=0**。详见报告第十一节。


### 1.3 任务进度

| 项 | 状态 |
|---|---|
| 环境（NVIDIA 4090 / CUDA 13.0 / torch 2.6.0 / fuxictr 2.0.1） | ✅ 可用 |
| **iPinYou 复现（FCNv1）** | ✅ **完成，AUC 78.4826（作者 78.4827）** |
| **KKBox 复现（FCNv1 / FCNv2）** | ✅ **完成，AUC 85.6695 / 85.5105** |
| 数据：iPinYou | ✅ h5 已生成，Σvocab=665,765 **与论文精确相符** |
| 数据：KKBox | ✅ h5 已生成，Σvocab=91,756 **与论文精确相符** |
| 数据：Criteo | ✅ 完成（h5 三份，Σvocab=910,747 与作者 39/39 相符） |
| 数据：Avazu | ⚠️ 已解压，行数待复核（首次统计受 IO 干扰） |
| E0 基线 8 种子 | 🔄 运行中（seed2025 完成，2026 进行中） |
| 基线模型（12 个） | ⏳ 代码已就位（FuxiCTR model_zoo），待接入 |
| 性能优化 | ⏳ 待开展（见 §五） |

---

## 二、目录结构

```
course_ctr_fcn_20261118/
├── code/
│   ├── FCN_repo/           作者原始仓库（零改动，复现基准）
│   ├── configs/            15 个 expid 配置
│   ├── configs_h5_author/  ★ 复现用配置（严格照抄作者日志超参）
│   └── src/                改进实现（LCA 参数化 + Tri-BCE 软权重）
├── tools/                  工具脚本
│   ├── run_expid_fixed.py  修正版训练入口（绕开官方脚本三处缺陷）
│   ├── h5conv.py           ★ CSV→h5 通用转换（含 vocab 双校验）
│   └── build_h5_any.py     ★ 按 dataset_config 驱动的转换脚本
├── report/                 课程设计报告 + ★ 复现报告.md
└── data/                   不入库（28 GB）
    ├── raw/                三数据集 zip
    ├── ipinyou_x1_h5/      h5 成品
    ├── kkbox_x1_h5/        h5 成品
    └── Criteo_x4_10_h5/    h5 成品
```

> 说明：`docs/`（过程草稿）与 `scripts/`（早期探索脚本）已移出版本控制，保持仓库精简；本地备份于仓库外。

---

## 三、关键事实（勿重复调研）

### 3.1 环境

```
加速卡 : 海光 DCU K500SM_AI (gfx928, 120 CU, 63.98 GiB)
DTK    : 26.04      必须先 source /opt/dtk-26.04/env.sh
PyTorch: 2.7.1+das.opt1.dtk2604
Python : 3.11.9（/usr/local/bin/python3）
框架   : fuxictr==2.0.1
必锁   : scikit-learn==1.4.2（≥1.5 移除 log_loss 的 eps 参数，第 2 轮后崩）
         numpy==1.25.0
cgroup : 6 核 CPU / 115 GB 内存
```

**本次复现环境（NVIDIA，2026-11-21）**：

```
加速卡 : NVIDIA GeForce RTX 4090 (24 GB, 24564 MiB)
驱动   : 580.82.07 / CUDA 13.0
PyTorch: 2.6.0+cu124
Python : 3.11（conda env fcn_cy）
框架   : fuxictr==2.0.1（与作者要求一致）
```

两套环境共用同一份模型代码，迁移只改 torch 安装方式（DTK 版 → CUDA 版），**代码零改动**，指标仍可复现到 1e-6。

**外网**（此前误判为无外网，已更正）：
```bash
source /root/private_data/.ai_user_info/ai_proxy
# http(s)_proxy -> 10.15.100.43:3120；no_proxy 含 ksai.scnet.cn
# 实测 github/pypi/huggingface 均 HTTP 200
```

### 3.2 各数据集超参**各不相同**（关键，不能一套配置通吃）

| 数据集 | batch | emb_dim | dropout | exp层 | lin层 | heads | batch_norm |
|---|---|---|---|---|---|---|---|
| **Criteo** | **8192** | 16 | 0.1 | 4 | 1 | 1 | **False** |
| **Avazu** | 10000 | 16 | **0** | 4 | 4 | **4** | True |
| **KKBox** | 10000 | **128** | 0.2 | 4 | **0** | 1 | True |
| **iPinYou** | 4096 | 16 | **0.3** | **0** | 4 | 2 | True |
| **ML-1M** | 4096 | 16 | 0.1 | **2** | **3** | 1 | True |

来源：`code/FCN_repo/config/model_config.yaml`。
反直觉点：KKBox 的 emb_dim=128；iPinYou 的 exp 层=0（无 ECN）；Avazu heads=4；Criteo 是唯一 batch_norm=False。

### 3.3 论文正文与作者实际实现**三处不一致**（一律以后者为准）

| 项目 | 论文正文 | 作者实现 |
|---|---|---|
| Criteo batch_size | 10000 | **8192**（日志 4477 batches 可反证） |
| 分桶规则 | `⌊log2(x)⌋` | `⌊(ln x)²⌋` |
| `x ≤ 2` 时 | 映射到 1 | 保留原值（0/1/2） |

### 3.4 数据集规模（论文 Table 2）

| 数据集 | 实例数 | 字段 | 特征数 | 我们的核对 |
|---|---|---|---|---|
| Criteo | 45,840,617 | 39 | 910,747 | ✅ 精确相符 |
| iPinYou | 19,495,974 | 16 | 665,765 | ✅ 精确相符 |
| KKBox | 7,377,418 | 13 | 91,756 | ✅ 精确相符 |
| Avazu | 40,428,967 | 24 | 3,750,999 | ⚠️ 待复核 |

---

## 四、数据集准备

```bash
source /root/private_data/.ai_user_info/ai_proxy
P=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
mkdir -p $P/data/raw && cd $P/data/raw

# 三个数据集（reczoo 官方发布版）
for ds in iPinYou_x1 KKBox_x1 Avazu_x4; do
  curl -sSL -o ${ds}.zip "https://huggingface.co/datasets/reczoo/${ds}/resolve/main/${ds}.zip"
done

# 解压
cd $P/data
unzip -q raw/iPinYou_x1.zip -d iPinYou
unzip -q raw/KKBox_x1.zip  -d KKBox_x1
unzip -q raw/Avazu_x4.zip  -d Avazu_x4
```

**压缩包很小**（iPinYou 208 MB / KKBox 963 MB / Avazu 1.3 GB），解压后 CSV 约 2 GB / 1 GB / 4.7 GB。

---

## 五、性能优化清单（当前主线）

目标：**在 6 核 / 115 GB 的小机器上跑起来，且不改变数值结果**。

| # | 问题 | 位置 | 现状 | 优化方向 |
|---|---|---|---|---|
| 1 | `convert_to_bucket` 逐元素 Python 调用 | `fuxictr.datasets.criteo` | 4.77 亿次调用，约占预处理 99% 时间 | 向量化 `np.floor(np.log(arr)**2)`，预期快 20–50× |
| 2 | `read_csv` 一次性全量载入 | `FeatureProcessor.read_csv` | `pd.read_csv(dtype=object)`，Criteo 峰值 45 GB | 改 `chunksize` 流式 |
| 3 | `preprocess` 新建列时旧列未释放 | `convert_to_bucket` 的 `.map()` | 峰值比预期高 5% | 原地替换 |
| 4 | 训练与重 IO 任务并行 | 操作纪律 | 实测把 ep7 从 260 s 拖到 **1266 s** | 串行：训练期间不做重 IO |

> **优化铁律**：任何改动必须先用 Criteo 已验的 `Σvocab = 910,747` 与 `I1 = 44` 做回归测试。
> 只要一个桶的归属变了，这两个数就对不上——**这是免费的精确判据**。

---

## 六、基线

论文要求 16 个基线。FuxiCTR 官方 `model_zoo` 提供 **12 个现成实现**（与作者同一框架）：

```
DCN(DCNv1) DCNv2 DeepFM WideDeep PNN xDeepFM AutoInt AFN
EDCN MaskNet EulerNet FinalMLP
```

缺 3 个：`FINAL`（SIGIR'23）、`RFM`（2024）、`DLF`（2025）——引用 reczoo Open Benchmarking 公开榜单值，报告中标注来源。

**超参策略**：每个基线使用该作者为该数据集给出的配置；仅对 FCN / DCNv1 / DCNv2 做 dropout 网格搜索。

---

## 七、跑实验

```bash
source /opt/dtk-26.04/env.sh
P=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
cd $P

# E0 基线（8 种子）
python3 tools/run_e0_baseline_v2.py \
  --repo code/FCN_repo --workdir . --data-root data \
  --seeds 2025,2026,2027,2028,2029,2030,2031,2032 \
  --exp-layers 4 --lin-layers 1 --epochs 30
```

`--epochs 30` 是安全上限（作者 19 轮是早停结果）；实测首种子 23 轮自然早停，未触及。

---

## 八、注意事项

1. **重跑转换前必须 `rm -f data/*/feature_map.json`**。残留的 feature_map 会让 `build_dataset` 打 `Skip rebuilding` 并静默跳过——「秒完成并报成功」但 h5 是空的。
2. **训练期间不要跑重 IO**（数行、解压、转 h5）。实测 `wc -l` 扫大文件会把每轮从 260 s 拖到 1266 s。
3. **作者 `checkpoints/` 下有两批日志，来自两个代码库**。同目录 `FCNv2_Criteo.log` 是旧实现（3ECN+3LCN，16,138,194 参数），**用 HEAD 代码复现不出**，不能作基准。基准是 `FCNv2_Criteo_016_cde6d2ff.log`（exp=4/lin=1，16,532,562）。

---

## 九、引用

```bibtex
@inproceedings{fcn2026,
  title     = {FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770854.3780177}
}
```
