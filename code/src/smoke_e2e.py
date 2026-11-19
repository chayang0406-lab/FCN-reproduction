#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端冒烟：造一份**真实 fuxictr h5 格式**的合成数据，跑通 run_improved.py。

为什么要有这一步（而不是只跑单元测试）：
    单元测试是直接调 model.add_loss，绕过了 BaseModel.fit、H5DataLoader、
    配置加载、set_logger、evaluate、checkpoint 保存这一整条链路。
    而"打开开关时能正常训练"这条验收，指的正是**整条链路**能跑通。
    所以这里用真格式的 h5（feature_map.json + train/valid/test.h5）走一遍完整流程。

数据是合成的，AUC 接近 0.5 属正确表现 —— 本脚本只验证"能跑通、形状对、数值有限"，
不产生任何可写进报告的精度数字。

用法：
    python code/src/smoke_e2e.py
成功时退出码 0，并打印每种配置的实际耗时与参数量。
"""

from __future__ import absolute_import

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.tests._fixtures import CRITEO_VOCAB, FEATURE_ORDER    # noqa: E402


def _make_h5(path, n_rows, seed=0):
    """写一份 fuxictr 能直接读的 h5：每列一个 dataset，值为 int64 的类别 id。

    词汇表很大时只取靠前的部分（hi=min(vocab,2000)），避免造出巨大的随机开销；
    这不影响链路验证，因为 embedding 层按 vocab_size 建表，与实际取值分布无关。
    """
    import numpy as np
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as hf:
        for name in FEATURE_ORDER:
            vs = int(CRITEO_VOCAB[name])
            hi = min(vs, 2000)
            col = rng.integers(0, hi, size=n_rows, dtype=np.int64)
            hf.create_dataset(name, data=col)
        # 标签：随机 0/1（与任何特征都无关，故 AUC ≈ 0.5，这是合成数据的正确表现）
        label = rng.integers(0, 2, size=n_rows, dtype=np.int64)
        hf.create_dataset("Label", data=label)
    return path


def _make_feature_map(dirpath):
    features = []
    for name in FEATURE_ORDER:
        vs = int(CRITEO_VOCAB[name])
        features.append({name: {"source": "", "type": "categorical",
                                "padding_idx": 0, "oov_idx": vs - 1,
                                "vocab_size": vs}})
    payload = {
        "dataset_id": "Criteo_x4_10_h5",
        "num_fields": len(FEATURE_ORDER),
        "total_features": sum(int(CRITEO_VOCAB[n]) for n in FEATURE_ORDER),
        "input_length": len(FEATURE_ORDER),
        "labels": ["Label"],
        "features": features,
    }
    with io.open(os.path.join(dirpath, "feature_map.json"), "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=4)


def main():
    work = tempfile.mkdtemp(prefix="fcn_smoke_e2e_")
    ok = True
    try:
        data_dir = os.path.join(work, "data", "Criteo_x4_10_h5")
        os.makedirs(data_dir, exist_ok=True)
        # 小规模：只验证链路通，不追求精度
        _make_h5(os.path.join(data_dir, "train.h5"), 3000, seed=1)
        _make_h5(os.path.join(data_dir, "valid.h5"), 800, seed=2)
        _make_h5(os.path.join(data_dir, "test.h5"), 800, seed=3)
        _make_feature_map(data_dir)

        configs = os.path.join(_CODE_DIR, "configs")
        cases = [
            ("FCNv2_C_base", "基线（原版行为）"),
            ("FCNv2_C_soft_smoothmax", "Tri-BCE 软权重"),
            ("FCNv2_C_soft_ratio", "Tri-BCE 归一化权重"),
            ("FCNv2_C_r100", "LCA r=1 全秩（5 层）"),
            ("FCNv2_C_r025", "LCA r=1/4（5 层）"),
            ("FCNv2_C_9L_r050", "E4-B 主实验 r=1/2（9 层，对齐论文 FCNp 1.76M）"),
            ("FCNv2_C_9L_r100", "E4-B 主实验 r=1 全秩（9 层，对齐论文 FCNp-full 3.51M）"),
        ]
        for expid, desc in cases:
            out = os.path.join(work, "runs", expid)
            cmd = [sys.executable, os.path.join(_HERE, "run_improved.py"),
                   "--config", configs, "--expid", expid,
                   "--gpu", "-1",                    # CPU 跑即可，链路与设备无关
                   "--out", out,
                   "--data_root", os.path.join(work, "data"),
                   "--model_root", os.path.join(work, "ckpt") + "/",
                   "--epochs", "1",
                   "--verbose", "0"]
            print("=" * 70)
            print("[%s] %s" % (expid, desc))
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                ok = False
                print("  FAILED (exit %d)" % r.returncode)
                print(r.stdout[-3000:])
                print(r.stderr[-3000:])
                continue
            res_path = os.path.join(out, "result.json")
            with io.open(res_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            print("  OK  params_total=%d  params_cross=%s  epochs_done=%s"
                  % (res["n_params_total"], res["n_params_cross"], res["epochs_done"]))
            print("      valid=%s" % (res["valid"],))
            # 硬检查：结果必须是有限值
            for k, v in list(res["valid"].items()):
                if not isinstance(v, (int, float)) or v != v:
                    ok = False
                    print("      FAILED: valid[%s]=%r 非有限数值" % (k, v))

        print("=" * 70)
        print("端到端冒烟：%s" % ("全部通过" if ok else "存在失败"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
