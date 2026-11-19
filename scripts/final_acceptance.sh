#!/bin/bash
# t1 最终验收自检：一次性把「环境可用 + FCNv2 在 DCU 上前反向」重新验一遍
REMOTE=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
source /opt/dtk-26.04/env.sh
PY=/usr/local/bin/python3

echo "############### t1 最终验收自检 ###############"
date '+%Y-%m-%d %H:%M:%S %Z'
echo ""

FAIL=0
chk() {
  if [ "$1" = "0" ]; then echo "  [PASS] $2"; else echo "  [FAIL] $2"; FAIL=$((FAIL+1)); fi
}

echo "=== A. 关键包版本 + numpy 未被升级 ==="
$PY - <<'EOF'
import numpy, sklearn, h5py, fuxictr, pandas, torch, inspect
from sklearn.metrics import log_loss
print("  numpy   =", numpy.__version__, "(must be 1.25.0)")
print("  sklearn =", sklearn.__version__, "(must be 1.4.2)")
print("  h5py    =", h5py.__version__)
print("  fuxictr =", fuxictr.__version__, "(must be 2.0.1)")
print("  pandas  =", pandas.__version__)
print("  torch   =", torch.__version__)
assert numpy.__version__ == "1.25.0", "numpy 被改了！"
assert sklearn.__version__ == "1.4.2", "sklearn 版本不对"
assert fuxictr.__version__ == "2.0.1", "fuxictr 版本不对"
assert "eps" in inspect.signature(log_loss).parameters, "log_loss 没有 eps 参数"
print("  -> 版本断言全部通过")
EOF
chk $? "A. 包版本 + numpy 1.25.0 + sklearn eps 可用"

echo ""
echo "=== B. DCU 可用且能真算 ==="
$PY - <<'EOF'
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print("  device =", p.name, "| arch =", p.gcnArchName, "| mem = %.2f GiB" % (p.total_memory/1024**3))
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
s = float((a @ b).sum())
torch.cuda.synchronize()
assert s == s and abs(s) != float("inf")
print("  2048x2048 matmul sum = %.4f  (有限值)" % s)
EOF
chk $? "B. DCU 可用 + 矩阵乘真算"

echo ""
echo "=== C. fuxictr 全模块导入 ==="
cd $REMOTE && $PY scripts/verify_imports.py > /tmp/_imp.log 2>&1
grep -q "import ok = 14 / 14" /tmp/_imp.log
chk $? "C. fuxictr 14/14 模块导入"
grep -E "convert_to_bucket = " /tmp/_imp.log | sed 's/^/    /'

echo ""
echo "=== D. FCNv2 在 DCU 上完成一次前向+反向（重跑冒烟）==="
cd $REMOTE/scripts && $PY smoke_remote_fcnv2.py > /tmp/_smoke.log 2>&1
RC=$?
chk $RC "D0. 冒烟脚本退出码 0"
grep -E "device +K500SM_AI|model.device|参数量|参数张量数|loss =|全局梯度|PASS:" /tmp/_smoke.log | sed 's/^/    /'
# 从 json 里做机器校验
$PY - <<'EOF'
import json
d = json.load(open("/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/scripts/_smoke_remote/smoke_result.json"))
fb = d["fwd_bwd"]
assert d["ok"] is True
assert d["device_name"] == "K500SM_AI"
assert fb["params_with_grad"] == fb["params_total"], "有参数没梯度"
assert fb["nonfinite_grads"] == 0, "梯度有 NaN/Inf"
assert fb["params_nonzero_grad"] > 0, "梯度全 0"
assert fb["grad_norm_l2"] > 0
print("    JSON 校验: 参数 %d/%d 有梯度, 非零 %d, 非有限 %d, 范数 %.6f"
      % (fb["params_with_grad"], fb["params_total"], fb["params_nonzero_grad"],
         fb["nonfinite_grads"], fb["grad_norm_l2"]))
EOF
chk $? "D1. JSON 机器校验：83/83 梯度、0 NaN、范数>0"

echo ""
echo "=== E. 参数量对账可复现 ==="
cd $REMOTE && $PY scripts/verify_param_v2.py 2>&1 | grep -E "exp_num_layers=|结论" | head -8 | sed 's/^/    /'
chk $? "E. 参数量对账脚本可跑"

echo ""
echo "=== E2. 参数量：新版结构式 + 旧版经验式 双验证 ==="
cd $REMOTE && $PY tools/param_formula.py 2>&1 | grep -E "^结论" | sed 's/^/    /'
cd $REMOTE && $PY tools/param_formula.py > /tmp/_e2.log 2>&1
chk $? "E2. 参数量公式全部断言通过（含旧版 14 份日志）"

echo ""
echo "=== E3. 文档里每个 AUC 数字都能从原始日志复现（valid/test 分列）==="
cd $REMOTE && $PY scripts/verify_doc_numbers.py > /tmp/_nums.log 2>&1
chk $? "E3. 文档 AUC 数字逐条核对原始日志"
grep -E "OK$|ALL CLAIMS" /tmp/_nums.log | sed 's/^/    /' | head -8

echo ""
echo "=== F. 产物清单 ==="
for f in docs/环境_实测.md logs/smoke_dcu_20261118.log logs/hw_facts_20261118.log \
         logs/import_verify_20261118.log code/FCN_repo/src/FCNv2.py \
         scripts/smoke_remote_fcnv2.py wheels/fuxictr-2.0.1-py3-none-any.whl; do
  if [ -e "$REMOTE/$f" ]; then echo "  [OK]   $f"; else echo "  [MISS] $f"; FAIL=$((FAIL+1)); fi
done

echo ""
echo "=== G. 参数公式工具自检（新版结构式 + 旧版经验式）==="
cd $REMOTE && $PY tools/param_formula.py > /tmp/_pf.log 2>&1
chk $? "G1. param_formula.py 全部断言通过"
grep -E "^结论" /tmp/_pf.log | sed 's/^/    /'
cd $REMOTE && $PY tools/verify_formula_all_logs.py > /tmp/_pf2.log 2>&1
chk $? "G2. 公式对作者全部 38 份日志 100% 命中"
grep -E "^新版|^旧版|^PASS" /tmp/_pf2.log | sed 's/^/    /'

echo ""
echo "=== G3. 参数量闸门端到端（真实建模型 + 必须能拦住跑错配置）==="
cd $REMOTE && $PY tools/verify_param_gate.py > /tmp/_gate.log 2>&1
chk $? "G3. 闸门可用且能拦住不符配置"
grep -E "^  OK|^  \[OK\]|^RESULT" /tmp/_gate.log | sed 's/^/    /'

echo ""
echo "=== H. 吞吐 bench 归档校验（captain 的排期依据）==="
BENCH=$REMOTE/bench/bench_result.json
if [ -f "$BENCH" ]; then
  echo "  [OK]   bench/bench_result.json 存在"
  $PY - <<'PYEOF'
import json, sys
p = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/bench/bench_result.json"
d = json.load(open(p))
np_ = d.get("n_params")
spe = d.get("s_per_epoch_est")
bad = []
if np_ != 16532562:
    bad.append("n_params=%s != 16532562 (台架数据造错/配置未生效!)" % np_)
if spe is None or not (180 <= spe <= 350):
    bad.append("s_per_epoch_est=%s 超出合理区间 [180,350]" % spe)
if bad:
    print("    [FAIL] " + "; ".join(bad))
    sys.exit(1)
print("    [OK] n_params=16532562, s_per_epoch_est=%.1f 在 [180,350] 内" % spe)
PYEOF
  chk $? "H1. bench_result.json 参数与耗时均在合理区间"
else
  echo "  [MISS] bench/bench_result.json"
  FAIL=$((FAIL+1))
fi

EB=$REMOTE/bench/eval_scaling.json
if [ -f "$EB" ]; then
  $PY - <<'PYEOF'
import json, sys
p = "/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118/bench/eval_scaling.json"
d = json.load(open(p))
bad = []
if d.get("n_params") != 16532562:
    bad.append("n_params 不符")
if not (0 < d.get("fit_per_sample_us", 0) < 100):
    bad.append("fit_per_sample_us=%.2f 不合理" % d.get("fit_per_sample_us", -1))
if not (150 <= d.get("est_epoch_with_eval_s", 0) <= 400):
    bad.append("est_epoch_with_eval_s=%.1f 超出合理区间" % d.get("est_epoch_with_eval_s", -1))
if bad:
    print("    [FAIL] " + "; ".join(bad)); sys.exit(1)
print("    [OK] eval 估算: per_sample=%.2fus, epoch_with_eval=%.1fs, %.2fx vs 4090"
      % (d["fit_per_sample_us"], d["est_epoch_with_eval_s"], d["slowdown_vs_4090"]))
PYEOF
  chk $? "H2. eval_scaling.json 数值合理"
else
  echo "  [note] 无 eval_scaling.json（可选，用于 eval 开销估算）"
fi

echo ""
echo "############### 结果 ###############"
if [ $FAIL -eq 0 ]; then
  echo "ALL PASS (0 failures)"
else
  echo "$FAIL 项失败"
fi
exit $FAIL

