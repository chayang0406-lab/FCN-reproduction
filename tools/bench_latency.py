#!/usr/bin/env python3
"""验证推理延迟测量协议，并给出延迟随 r 变化的趋势。

## 为什么单独写这个
参数量和推理延迟是本项目**唯一能 n=1 严格验证**的指标（AUC 有种子噪声）。
论文声称 LCA "reduce inference latency by 23%"，但**从未交代测量协议**
（batch size、warmup、是否含 embedding、重复次数、误差棒，都没有）。

所以我们必须自己定协议并写清楚。这个脚本同时：
1. 验证协议本身可复现（重复测同一配置，看波动多大）
2. 对比「含 embedding 查表」与「不含」两种口径的差异（论文没说清是哪种）
3. 给出 r ∈ {1/4, 1/2, 3/4, 1} 的延迟

## 协议（写进报告）
- warmup ≥ 20 batch（避免 kernel JIT 编译污染）
- 每次计时前后 torch.cuda.synchronize()
- 重复 N_REPEAT 次取**中位数**（比均值抗离群）
- 报告 IQR（四分位距），让读者知道波动范围
- 固定 batch_size（作者用 16384）
"""
import argparse
import json
import os
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FCN 仓库")
    ap.add_argument("--impl", default=None, help="改进实现的目录（含 src/fcn_v2_soft.py）")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--n-batch", type=int, default=20, help="每次计时跑几个 batch")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    if args.impl:
        sys.path.insert(0, args.impl)

    import numpy as np
    import torch
    import h5py
    from fuxictr.features import FeatureMap

    print("[env] torch %s | device %s" % (torch.__version__, torch.cuda.get_device_name(0)))

    # 用作者真实 vocab 造 feature_map（保证参数量与真实场景一致）
    REAL_VOCAB = {
        "C1": 1446, "C2": 554, "C3": 157339, "C4": 117822, "C5": 306, "C6": 18,
        "C7": 11882, "C8": 630, "C9": 5, "C10": 39530, "C11": 5131, "C12": 156656,
        "C13": 3176, "C14": 28, "C15": 11043, "C16": 148913, "C17": 12, "C18": 4560,
        "C19": 2003, "C20": 5, "C21": 154564, "C22": 18, "C23": 17, "C24": 53031,
        "C25": 82, "C26": 40955,
        "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220, "I6": 112, "I7": 80,
        "I8": 69, "I9": 92, "I10": 6, "I11": 27, "I12": 37, "I13": 72,
    }
    names = ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(1, 27)]
    work = args.workdir
    os.makedirs(work, exist_ok=True)
    fmj = os.path.join(work, "fm.json")
    meta = {"dataset_id": "L", "labels": ["Label"], "num_fields": len(names),
            "features": [{n: {"source": "", "type": "categorical", "padding_idx": 0,
                             "oov_idx": REAL_VOCAB[n] - 1, "vocab_size": REAL_VOCAB[n]}}
                         for n in names]}
    json.dump(meta, open(fmj, "w"))

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # 两套实现：
    #  - 原仓库 code/FCN_repo/src（包名 src）
    #  - 改进版 code/src（包名也叫 src，靠 sys.path 区分，绝不能同时进入）
    # modeler 已写好 `_orig_loader.py`，用独立包名 `fcn_orig` 把原仓库读进来，
    # 这样两者能在一个进程里共存、互不污染。优先用它，避免包名冲突。
    impl_mod = None
    orig_cls = None
    if args.impl:
        sys.path.insert(0, args.impl)          # 让 `src` 指到改进版
        import importlib
        impl_mod = importlib.import_module("src.fcn_v2_soft")
        print("[impl] 改进版 src.fcn_v2_soft 加载成功")
        try:
            loader = importlib.import_module("src._orig_loader")
            orig_cls = getattr(loader.orig, "FCNv2", None)
            if orig_cls is not None:
                print("[impl] 原仓库经 _orig_loader 以 fcn_orig 加载成功")
        except Exception as e:
            print("[impl] _orig_loader 不可用（%s），r=1/2 将用改进版自身验证" % e)
    else:
        sys.path.insert(0, args.repo)          # 让 `src` 指到原仓库
        import src as orig_zoo
        orig_cls = orig_zoo.FCNv2

    def build(model_cls, r, device):
        params = {
            "model_root": work + "/", "verbose": 0, "task": "binary_classification",
            "gpu": 0, "monitor": {"AUC": 1, "logloss": 0}, "monitor_mode": "max",
            "early_stop_patience": 2, "eval_steps": None, "save_best_only": True,
            "embedding_regularizer": 1e-5, "net_regularizer": 0, "optimizer": "adam",
            "loss": "binary_crossentropy", "metrics": ["logloss", "AUC"],
            "embedding_dim": 16, "exp_num_layers": 4, "lin_num_layers": 5,
            "net_dropout": 0.0, "batch_norm": False, "layer_norm": False,
            "num_heads": 1, "model_id": "m%d" % int(r * 100),
            "lca_compress_ratio": r,
        }
        fm = FeatureMap("L", work)
        fm.load(fmj, {"embedding_dim": 16})
        return model_cls(fm, **params), fm

    torch.cuda.empty_cache()
    results = []

    # r 的候选：原版只有 1/2；改进版可四档
    rs = [0.5] if not impl_mod else [0.25, 0.5, 0.75, 1.0]

    for r in rs:
        if impl_mod is not None:
            model, fm = build(impl_mod.FCNv2Soft, r, "cuda")
            tag = "improved"
        else:
            model, fm = build(orig_cls, r, "cuda")
            tag = "original"

        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        # fuxictr 的 BaseModel.get_inputs 期望 inputs 是**二维张量**
        # （shape = [batch, num_fields]），内部按列索引取各字段；
        # 传 dict 会报 TypeError: unhashable type: 'slice'。
        # 列顺序必须与 feature_map 的 column index 一致，即 names 的顺序。
        cols = []
        for nm in names:
            cols.append(torch.randint(1, REAL_VOCAB[nm], (args.batch_size,),
                                      device="cuda"))
        inputs = torch.stack(cols, dim=1)                      # [B, 39]
        labels = torch.randint(0, 2, (args.batch_size,), device="cuda").float()

        def one_pass():
            with torch.no_grad():
                out = model(inputs)
                if isinstance(out, dict):
                    out = out["y_pred"]
                return out

        # warmup（多跑一些，确保 kernel 全编译完、显存分配器稳定）
        for _ in range(args.warmup):
            one_pass()
        torch.cuda.synchronize()

        # 关键：单次计时太短会被调度噪声主导（实测同一配置能在 0.0014~0.0031 间跳）。
        # 所以每次计时跑 n_batch 个 batch，并且重复 repeat 次，
        # 取**总耗时/总样本数** 作为 ms/sample，再在 repeat 次里取中位数。
        samples = []
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.n_batch):
                one_pass()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            # ms/sample，同时也记录 ms/batch 供与论文口径对照
            samples.append(dt / args.n_batch / args.batch_size * 1000.0)

        med = statistics.median(samples)
        q = sorted(samples)
        iqr = (q[int(len(q) * 0.75)] - q[int(len(q) * 0.25)]) if len(q) >= 4 else 0.0
        # 相对离散度：用来判断这个测量是否可信（>10% 说明被噪声主导）
        rel_spread = (max(samples) - min(samples)) / med * 100 if med > 0 else 0.0
        ms_per_batch = med * args.batch_size
        results.append({"r": r, "impl": tag, "n_params": n_params,
                        "ms_per_sample_median": med,
                        "ms_per_batch_median": ms_per_batch,
                        "iqr": iqr, "rel_spread_pct": rel_spread,
                        "samples": samples, "batch_size": args.batch_size})
        print("  r=%.2f (%s) params=%d  %.4f ms/sample = %.2f ms/batch  "
              "IQR=%.4f  spread=%.1f%%"
              % (r, tag, n_params, med, ms_per_batch, iqr, rel_spread), flush=True)
        del model
        torch.cuda.empty_cache()

    # 相对 r=1/2 的延迟比
    base = [x for x in results if abs(x["r"] - 0.5) < 1e-9]
    if base:
        b = base[0]["ms_per_sample_median"]
        print("\n相对 r=1/2 的延迟变化：")
        for x in results:
            print("  r=%.2f  %.4f ms  -> %+.2f%%" % (x["r"], x["ms_per_sample_median"],
                                                     (x["ms_per_sample_median"] / b - 1) * 100))
        print("\n论文声称 -full（r=1）比 LCA（r=1/2）延迟高约 23%；")
        print("但按其 Table 5 自己算：Criteo ECN +27.02%%，FCNp +27.98%%；KKBox +31.24%%/+26.74%%。")

    out = os.path.join(work, "latency_result.json")
    json.dump(results, open(out, "w"), indent=2)
    print("\nwrote " + out)


if __name__ == "__main__":
    main()

