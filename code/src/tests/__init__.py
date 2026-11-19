# -*- coding: utf-8 -*-
"""`src.tests` 包：放改进版的单元测试。

不用 `__init__.py` 暴露任何东西，测试文件按需直接导入，
避免 pytest 在收集阶段 import 测试模块时意外拉起 fuxictr。
"""
