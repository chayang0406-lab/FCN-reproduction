#!/usr/bin/env bash
# 在远程 DCU 上建立改进代码目录并跑全部验证。
# 用法：bash remote_verify_improve.sh <项目根> [--with-gpu]
#
# 注意：这里**不能**在 source env.sh 之前 `set -u`。
#       /opt/dtk-26.04/env.sh 里引用了若干可能未定义的变量，
#       开了 nounset 会让 shell 在 source 阶段直接静默退出（exit 1、无任何输出），
#       极难排查。改为 source 之后再开。
set +u
R="${1:-/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118}"
WITH_GPU="${2:-}"
cd "$R" || exit 1
source /opt/dtk-26.04/env.sh >/dev/null 2>&1
set -u
PY=/usr/local/bin/python3

echo "############ 0. 环境核对 ############"
$PY - <<'PYEOF'
import sys, torch, numpy, sklearn, fuxictr
print("python ", sys.version.split()[0])
print("torch  ", torch.__version__, "| cuda avail", torch.cuda.is_available())
print("numpy  ", numpy.__version__)
print("sklearn", sklearn.__version__)
print("fuxictr", fuxictr.__version__)
if torch.cuda.is_available():
    print("device ", torch.cuda.get_device_name(0))
    print("gcnArch", getattr(torch.cuda.get_device_properties(0), "gcnArchName", "n/a"))
PYEOF

echo
echo "############ 1. 布局核对 ############"
ls -1 "$R/code" 2>/dev/null
echo "--- code/src (改进版) ---"; ls -1 "$R/code/src" 2>/dev/null
echo "--- code/src/tests ---";     ls -1 "$R/code/src/tests" 2>/dev/null
echo "--- code/configs ---";       ls -1 "$R/code/configs" 2>/dev/null
echo "--- code/FCN_repo/src (原仓库, 只读基线) ---"; ls -1 "$R/code/FCN_repo/src" 2>/dev/null
echo "--- 原仓库 FCNv2.py 的 sha256 (用于证明未被改动) ---"
sha256sum "$R/code/FCN_repo/src/FCNv2.py" 2>/dev/null

echo
echo "############ 2. LCA 参数量推导与自检 ############"
cd "$R/code" || exit 1
PYTHONPATH="$R/code" $PY -X utf8 src/tools_check_lca.py
echo "tools_check_lca.py exit=$?"

echo
echo "############ 3. 逐位一致测试 ############"
PYTHONPATH="$R/code" $PY -X utf8 src/tests/test_bit_exact.py
BITEXACT_RC=$?
echo "test_bit_exact.py exit=$BITEXACT_RC"

echo
echo "############ 4. 变异测试（证明断言有判别力） ############"
PYTHONPATH="$R/code" $PY -X utf8 src/tests/test_mutation.py
MUT_RC=$?
echo "test_mutation.py exit=$MUT_RC"

echo
echo "############ 5. 软权重与压缩率性质测试 ############"
PYTHONPATH="$R/code" $PY -X utf8 src/tests/test_soft_weight.py
PROP_RC=$?
echo "test_soft_weight.py exit=$PROP_RC"

echo
echo "############ 6. 端到端冒烟（合成 h5，CPU） ############"
PYTHONPATH="$R/code" $PY -X utf8 src/smoke_e2e.py
E2E_RC=$?
echo "smoke_e2e.py exit=$E2E_RC"

echo
echo "############ 7. GPU(=DCU) 上前反向验证 ############"
if [ "$WITH_GPU" = "--with-gpu" ]; then
R="$R" PYTHONPATH="$R/code" $PY - <<'PYEOF'
import os, sys, tempfile, shutil, json
R = os.environ["R"]
sys.path.insert(0, os.path.join(R, "code"))
import torch
from src.tests import _fixtures as fx
from src.tests.test_bit_exact import _build_pair
from src import fcn_v2_soft

out = {}
tmp = tempfile.mkdtemp(prefix="fcngpu_")
try:
    mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
              net_dropout=0.1, batch_norm=False, layer_norm=False,
              num_heads=1, learning_rate=1e-3)
    # 用 gpu=0 建模型（_build_pair 里写死 gpu=-1，这里单独建一次）
    from fuxictr.pytorch.torch_utils import seed_everything
    d = tempfile.mkdtemp(prefix="d_", dir=tmp)
    fx.write_feature_map_json(d)
    fm = fx.load_feature_map(d)
    kwargs = fx.base_kwargs(tmp); kwargs["gpu"] = 0
    seed_everything(seed=2025)
    m = fcn_v2_soft.FCNv2Soft(fm, tri_bce_mode="smoothmax", tri_bce_beta=100.0,
                              lca_compress_ratio=0.5, **dict(mk, **kwargs))
    print("model.device =", m.device)
    print("first param device =", next(m.parameters()).device)
    batch = fx.make_batch(fm, batch_size=4096, seed=1).to(m.device)
    m.train()
    loss = m.add_loss(batch)
    loss.backward()
    n_tot = n_grad = n_nonzero = n_bad = 0
    gnorm_sq = 0.0
    for name, p in m.named_parameters():
        if not p.requires_grad: continue
        n_tot += 1
        g = p.grad
        if g is None: continue
        n_grad += 1
        if not torch.isfinite(g).all(): n_bad += 1
        gn = float(g.norm())
        if gn > 0: n_nonzero += 1
        gnorm_sq += gn ** 2
    out = dict(device=str(m.device), loss=float(loss),
               params_total=n_tot, params_with_grad=n_grad,
               params_nonzero_grad=n_nonzero, params_nonfinite_grad=n_bad,
               global_grad_l2=gnorm_sq ** 0.5,
               n_params=sum(p.numel() for p in m.parameters()))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(out, indent=2))
ok = (out["params_with_grad"] == out["params_total"]
      and out["params_nonzero_grad"] == out["params_total"]
      and out["params_nonfinite_grad"] == 0
      and out["loss"] == out["loss"])
print("GPU_SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
GPU_RC=$?
echo "gpu smoke exit=$GPU_RC"
else
  echo "skipped (pass --with-gpu to run)"
  GPU_RC=0
fi

echo
echo "############ 汇总 ############"
echo "bit_exact=$BITEXACT_RC mutation=$MUT_RC property=$PROP_RC e2e=$E2E_RC gpu=$GPU_RC"
if [ "$BITEXACT_RC" -eq 0 ] && [ "$MUT_RC" -eq 0 ] && [ "$PROP_RC" -eq 0 ] && [ "$E2E_RC" -eq 0 ] && [ "$GPU_RC" -eq 0 ]; then
  echo "ALL_PASS"
  exit 0
else
  echo "SOME_FAILED"
  exit 1
fi
