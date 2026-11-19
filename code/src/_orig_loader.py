# -*- coding: utf-8 -*-
"""把原仓库 `code/FCN/src/` 当作**只读**基线，以独立包名 `fcn_orig` 加载进来。

为什么要这么绕一圈（而不是直接 `import src`）：
    本项目自己的改进代码放在 `code/src/`，包名也叫 `src`。
    如果两个包同名，Python 的模块缓存会让后加载的覆盖先加载的，
    实验时到底跑的是"原版"还是"改进版"就说不清了 —— 这是 A/B 对照里最致命的坑。
    所以这里用 importlib 以 `fcn_orig` 这个名字把原仓库加载成另一个包，
    两个包在同一个进程里共存、互不污染。

硬约束：**本文件只读原仓库文件，绝不写入、绝不改动原仓库任何字节。**
（原仓库要保持纯净，作为可对照的基线。）

加载方式说明：
    原仓库 `src/__init__.py` 里写的是相对导入 `from .ECN import *`，
    只把它所在目录塞进 sys.path 是import不到的（因为父目录才是包边界）。
    正确做法是给 spec 指定 `submodule_search_locations=[src_dir]`，
    这样 `fcn_orig` 就被当成一个包，相对导入才能正常解析。
"""

from __future__ import absolute_import

import os
import sys
import importlib.util

__all__ = ["orig", "orig_module", "ORIG_SRC_DIR"]

# 本文件位置：<项目>/code/src/_orig_loader.py  →  项目根 = 上两级
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)                 # <项目>/code
_PROJECT_DIR = os.path.dirname(_CODE_DIR)          # <项目>

# 候选路径：优先环境变量显式指定，其次按"本地 / 远程 DCU"两种已知布局依次探测。
# 远程 DCU 的项目根是 /public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
_CANDIDATES = [
    os.environ.get("FCN_ORIG_SRC", ""),
    os.path.join(_CODE_DIR, "FCN", "src"),                     # 本地：code/FCN/src
    os.path.join(_CODE_DIR, "FCN_repo", "src"),                # 远程：code/FCN_repo/src
    "/public/home/zengzhicun/user/huanghaijun/projects/"
    "course_ctr_fcn_20261118/code/FCN_repo/src",               # 远程兜底（绝对路径）
]


def _find_orig_src_dir():
    """返回原仓库 src 目录的绝对路径；找不到就报错并列出探测过的路径。"""
    tried = []
    for cand in _CANDIDATES:
        if not cand:
            continue
        cand = os.path.abspath(os.path.expanduser(cand))
        tried.append(cand)
        # 必须同时存在 __init__.py 与 FCNv2.py，才算"就是那个 src"
        if os.path.isfile(os.path.join(cand, "__init__.py")) and \
           os.path.isfile(os.path.join(cand, "FCNv2.py")):
            return cand
    raise RuntimeError(
        "找不到原仓库 src 目录。可用环境变量 FCN_ORIG_SRC 显式指定。\n"
        "已尝试以下路径：\n  " + "\n  ".join(tried)
    )


ORIG_SRC_DIR = _find_orig_src_dir()


def _load():
    """以 `fcn_orig` 为包名加载原仓库 src；重复调用返回同一个模块对象。"""
    if "fcn_orig" in sys.modules:
        return sys.modules["fcn_orig"]
    init_py = os.path.join(ORIG_SRC_DIR, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "fcn_orig", init_py, submodule_search_locations=[ORIG_SRC_DIR]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法为 %s 构造 import spec" % init_py)
    module = importlib.util.module_from_spec(spec)
    # 先注册再 exec：这样包内部的相对导入（from .ECN import *）才能找到 fcn_orig
    sys.modules["fcn_orig"] = module
    spec.loader.exec_module(module)
    return module


# 原仓库 src 包本体（已执行 __init__.py，属性名与 `import src` 完全一致）
orig = _load()


def orig_module(name):
    """取原仓库某个子模块，例如 orig_module("FCNv2") 拿 Exponential2LinearCrossNetwork 等类。"""
    return importlib.import_module("fcn_orig." + name)
