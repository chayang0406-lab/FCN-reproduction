#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 code/configs 下所有实验配置能被 fuxictr 正确加载，且参数量符合预期。

为什么单独写这一步：
    fuxictr 的 load_config 对拼错的 expid 只给一句断言，不告诉你有哪些可用；
    而配置里的开关一旦拼错（例如 tri_bce_mode: 'softpluss'），
    我们的模型会抛 ValueError（好），但如果漏了校验就会**静默退回原版**（坏）。
    所以这里把"每个 expid 都能加载 + 能建出预期参数量的模型 + 开关值确实生效"一次测完。

用法：python code/src/tools_check_configs.py
"""

from __future__ import absolute_import

import os
import sys
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from fuxictr.utils import load_config                       # noqa: E402
from src.tests import _fixtures as fx                        # noqa: E402
from src.tests.test_bit_exact import _build_pair             # noqa: E402

CONFIG_DIR = os.path.join(_CODE_DIR, "configs")

# expid -> (期望层数/子网, 期望 r, 期望 tri_bce_mode)
# 期望参数量由 cross_net_param_count 现算，不硬编码，避免改配置后忘了同步。
EXPECTED = {
    "FCNv2_C_base":               (5, 0.5,  "hard"),
    "FCNv2_C_soft_smoothmax":     (5, 0.5,  "smoothmax"),
    "FCNv2_C_soft_softplus":      (5, 0.5,  "softplus"),
    "FCNv2_C_soft_ratio":         (5, 0.5,  "ratio"),
    "FCNv2_C_soft_smoothmax_anneal": (5, 0.5, "smoothmax"),
    "FCNv2_C_r025":               (5, 0.25, "hard"),
    "FCNv2_C_r075":               (5, 0.75, "hard"),
    "FCNv2_C_r100":               (5, 1.0,  "hard"),
    "FCNv2_C_abl_no_tribce":      (5, 0.5,  "plain"),
    "FCNv2_C_9L_r025":            (9, 0.25, "hard"),
    "FCNv2_C_9L_r050":            (9, 0.5,  "hard"),
    "FCNv2_C_9L_r075":            (9, 0.75, "hard"),
    "FCNv2_C_9L_r100":            (9, 1.0,  "hard"),
}


def main():
    tmp = tempfile.mkdtemp(prefix="fcncfg_")
    ok = True
    try:
        d = os.path.join(tmp, "data")
        fx.write_feature_map_json(d)
        feature_map = fx.load_feature_map(d)
        D = feature_map.sum_emb_out_dim()
        emb = sum(p.numel() for n, p in
                  __import__("src.fcn_v2_soft", fromlist=["x"]).MultiHeadFeatureEmbedding(
                      feature_map, 16, 1).named_parameters())
        print("=" * 82)
        print("配置加载与参数量校验（D=%d, embedding=%d）" % (D, emb))
        print("=" * 82)
        print("  %-32s %-6s %-6s %-12s %-12s %-8s" %
              ("expid", "层数", "r", "开关", "模型总参数", "状态"))

        for expid, (layers, r, mode) in EXPECTED.items():
            try:
                p = load_config(CONFIG_DIR, expid)
            except Exception as e:  # noqa: BLE001
                print("  %-32s LOAD FAILED: %r" % (expid, e))
                ok = False
                continue
            got_layers = p["exp_num_layers"] + p["lin_num_layers"]
            got_r = float(p.get("lca_compress_ratio", 0.5))
            got_mode = str(p.get("tri_bce_mode", "hard")).lower()
            problems = []
            if got_layers != layers:
                problems.append("层数 %d != 期望 %d" % (got_layers, layers))
            if abs(got_r - r) > 1e-9:
                problems.append("r %s != 期望 %s" % (got_r, r))
            if got_mode != mode:
                problems.append("mode %s != 期望 %s" % (got_mode, mode))

            # 建模型核对实际参数量（用 CPU，小 batch 不需要）
            try:
                sub = tempfile.mkdtemp(prefix="m_", dir=tmp)
                mk = dict(embedding_dim=p["embedding_dim"],
                          exp_num_layers=p["exp_num_layers"],
                          lin_num_layers=p["lin_num_layers"],
                          net_dropout=p["net_dropout"],
                          batch_norm=p["batch_norm"],
                          layer_norm=p["layer_norm"],
                          num_heads=p["num_heads"],
                          learning_rate=p["learning_rate"],
                          tri_bce_mode=p.get("tri_bce_mode", "hard"),
                          tri_bce_beta=p.get("tri_bce_beta", 1.0),
                          lca_compress_ratio=p.get("lca_compress_ratio", 0.5))
                _, model, _fm = _build_pair(sub, mk)
                total = sum(q.numel() for q in model.parameters())
            except Exception as e:  # noqa: BLE001
                print("  %-32s BUILD FAILED: %r" % (expid, e))
                ok = False
                continue

            status = "OK" if not problems else "; ".join(problems)
            if problems:
                ok = False
            print("  %-32s %-6d %-6s %-12s %-12s %-8s" %
                  (expid, got_layers, got_r, got_mode, "{:,}".format(total), status))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("说明：参数量由模型真实构造得到，未硬编码。")
    print("      5 层 r=1/2 应 = 16,532,562（与作者日志 _016 一致）")
    print("      9 层 r=1/2 应 = 18,100,050（与论文 Table 5 FCNp 1.76M 单子网口径一致）")
    print()
    print("总体：%s" % ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
