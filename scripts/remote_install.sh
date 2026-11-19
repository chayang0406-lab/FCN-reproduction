#!/bin/bash
# 离线安装 FCN 依赖（DCU 机器无外网，只能 --no-index）
set -x
REMOTE=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
source /opt/dtk-26.04/env.sh
PY=/usr/local/bin/python3

echo "=== 0. before: numpy version (must stay 1.25.0) ==="
$PY -c "import numpy; print('numpy', numpy.__version__)"

echo "=== 1. offline install ==="
$PY -m pip install --no-index --find-links=$REMOTE/wheels \
    fuxictr==2.0.1 \
    scikit-learn==1.5.2 \
    h5py==3.10.0 \
    joblib==1.4.2 \
    polars==1.0.0 \
    keras_preprocessing==1.1.2 2>&1 | tail -40

echo "=== 2. after: numpy version (must STILL be 1.25.0) ==="
$PY -c "import numpy; print('numpy', numpy.__version__)"

echo "=== 3. versions ==="
$PY - <<'EOF'
import numpy, pandas, scipy, sklearn, h5py, yaml, tqdm, joblib, polars
print("numpy   ", numpy.__version__)
print("pandas  ", pandas.__version__)
print("scipy   ", scipy.__version__)
print("sklearn ", sklearn.__version__)
print("h5py    ", h5py.__version__)
print("PyYAML  ", yaml.__version__)
print("tqdm    ", tqdm.__version__)
print("joblib  ", joblib.__version__)
print("polars  ", polars.__version__)
import keras_preprocessing
print("keras_preprocessing", keras_preprocessing.__version__)
import fuxictr
print("fuxictr ", fuxictr.__version__)
EOF

