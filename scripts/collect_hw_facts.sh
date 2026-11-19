#!/bin/bash
# 记录硬件与软件栈实测证据
REMOTE=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
source /opt/dtk-26.04/env.sh

echo "############ 采集时间 ############"
date '+%Y-%m-%d %H:%M:%S %Z'

echo ""
echo "############ hy-smi ############"
which hy-smi
hy-smi 2>&1 | head -40

echo ""
echo "############ rocm-smi ############"
which rocm-smi
rocm-smi 2>&1 | head -40

echo ""
echo "############ rocm-smi --showproductname ############"
rocm-smi --showproductname 2>&1 | head -20

echo ""
echo "############ CPU / cgroup ############"
nproc
echo "--- cpu.max ---"
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "no cpu.max"
echo "--- memory.max ---"
cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "no memory.max"
echo "--- free -g ---"
free -g

echo ""
echo "############ 磁盘 ############"
df -h $REMOTE

echo ""
echo "############ Python / 关键包版本 ############"
/usr/local/bin/python3 -V
/usr/local/bin/python3 -m pip list 2>/dev/null | grep -iE '^(torch|numpy|pandas|scipy|scikit-learn|h5py|PyYAML|tqdm|joblib|polars|keras|fuxictr|pyarrow) '

echo ""
echo "############ torch 设备详情 ############"
/usr/local/bin/python3 - <<'EOF'
import torch
print("torch.__version__      =", torch.__version__)
print("torch.version.hip      =", getattr(torch.version, "hip", None))
print("torch.version.cuda     =", getattr(torch.version, "cuda", None))
print("cuda.is_available()    =", torch.cuda.is_available())
print("device_count()         =", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print("name                   =", p.name)
print("gcnArchName            =", getattr(p, "gcnArchName", "n/a"))
print("total_memory GiB       = %.2f" % (p.total_memory / 1024**3))
print("multi_processor_count  =", p.multi_processor_count)
print("major.minor            =", p.major, p.minor)

# 一次最小的设备算子验证
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = (a @ b).sum()
torch.cuda.synchronize()
print("matmul on DCU sum()    =", float(c))
print("cuda memory allocated  = %.2f MiB" % (torch.cuda.memory_allocated()/1024**2))
EOF

