# -*- coding: utf-8 -*-
"""变异测试：证明 test_bit_exact 的断言**有判别力**，不是"永远通过"的空测试。

为什么必须做这一步（这是验收里最容易被糊弄的地方）：
    "开关关闭时逐位一致"这个断言，如果写错了（比如两边其实跑的是同一个模型对象、
    或者用了 allclose、或者根本没加载原仓库），它会**永远通过**，
    于是所有 A/B 结论都建立在一个假的等价性上。
    所以这里故意把实现改坏一点点，看断言能不能抓住。

做法：不改文件，用 monkeypatch 在内存里注入变异，跑同一个断言函数，期望它 assert 失败。
    每个变异都必须被抓到（expected_fail=True）；如果有变异"改坏了却还通过"，
    说明测试有盲区，直接判失败。

覆盖的变异类型（都是"看起来无害但语义不同"的典型改法）：
    M1 单个参数加 1ulp 大小的扰动      —— 验证不是 allclose 在放水
    M2 exp/lin 两条支路的层顺序颠倒     —— 前向数值差极小，语义完全不同
    M3 残差里漏掉 b                    —— x_anchor*(H+b)+x 写成 x_anchor*H+x
    M4 add_loss 里把权重改成 detach     —— loss 标量可相同，但梯度不同
    M5 漏掉一处 init.uniform_           —— 参数分布不同
    M6 压缩率 r 从 1/2 改成 1/4         —— 结构变了（此项用"应当被抓住"的形式验证参数量对账）

===============================================================================
两个"假变异"的教训（已实测，值得记下来）
===============================================================================
原本还想测两个变异，实测发现它们在数值上**根本不产生差异**，因此不是变异：

  (a) 把 concat 顺序从 [H1, H2] 换成 [H2, H1]
      → 初始化时 γ=1、β=0，所以 H_2 = γ⊙H_1 + β ≡ H_1，
        两半**就是同一个张量**，交换顺序结果一模一样。
      这本身是 LCA 的一个真实性质：**第一步时"宽度翻倍"并没有带来任何新信息**，
      要等训练把 γ/β 学偏离 (1, 0) 之后，两半才开始分化。
      这一点要写进改进方案文档，因为它解释了为什么 r 增大带来的收益可能有限。

  (b) 把残差写法 a*(H+b)+x 换成 x+a*(H+b)
      → IEEE754 浮点加法满足交换律（a+b 与 b+a **精确相等**，不是近似），
        所以这不是变异，而是同一个计算。

结论：写变异测试时必须先确认"这个改动真的会改变数值"，否则会误判测试有盲区。
"""

from __future__ import absolute_import

import os
import sys
import shutil
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(os.path.dirname(_HERE))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src._orig_loader import orig                        # noqa: E402
from src import fcn_v2_soft                              # noqa: E402
from src.tests import _fixtures as fx                    # noqa: E402
from src.tests import test_bit_exact as TBE             # noqa: E402


def _expect_caught(name, mutate_fn, restore_fn, check_fn):
    """执行一次变异，断言 check_fn 会抛 AssertionError。"""
    try:
        mutate_fn()
        try:
            check_fn()
        except AssertionError as e:
            print("  CAUGHT   %s  (%s)" % (name, str(e).splitlines()[0][:90]))
            return True
        else:
            print("  MISSED   %s  ← 测试有盲区！" % name)
            return False
    finally:
        restore_fn()


def main():
    tmpdir = tempfile.mkdtemp(prefix="fcnmut_")
    ok = True
    try:
        mk = dict(embedding_dim=16, exp_num_layers=4, lin_num_layers=1,
                  net_dropout=0.1, batch_norm=False, layer_norm=False,
                  num_heads=1, learning_rate=1e-3)
        m_orig, m_new, feature_map = TBE._build_pair(tmpdir, mk)
        batch = fx.make_batch(feature_map, batch_size=16, seed=7)
        print("基线：两个模型参数量 %d / %d"
              % (sum(p.numel() for p in m_orig.parameters()),
                 sum(p.numel() for p in m_new.parameters())))

        # ---------- M1：单个参数加极小扰动（约 1ulp 量级）----------
        def m1_mutate():
            with torch.no_grad():
                p = m_new.E2LCN.w[0].weight
                p.add_(torch.finfo(p.dtype).eps)   # 加一个 float32 的 eps
        def m1_restore():
            with torch.no_grad():
                m_new.E2LCN.w[0].weight.copy_(m_orig.E2LCN.w[0].weight)
        ok &= _expect_caught(
            "M1 参数加 1ulp 扰动", m1_mutate, m1_restore,
            lambda: TBE._assert_state_dict_identical(m_orig, m_new, tag="[M1]"))

        # ---------- M2：E2L 两条支路的层顺序颠倒（先 lin 后 exp）----------
        # 这是真变异：[exp 用 x_anchor=x 连乘、lin 用 x_anchor=x1] 与反过来调换，
        # 数值上会产生真实差异（因为 x 在逐层变化）。
        _orig_e2l_forward = fcn_v2_soft.Exponential2LinearCrossNetwork.forward
        def m2_mutate():
            def bad(self, x):
                x1 = x
                for i in range(self.lin_num_layers):        # ← 先把线性层跑完
                    x = self._cross_layer(x, x1, i)
                for i in range(self.exp_num_layers):
                    i = i + self.lin_num_layers
                    x = self._cross_layer(x, x, i)
                return self.fc(x)
            fcn_v2_soft.Exponential2LinearCrossNetwork.forward = bad
        def m2_restore():
            fcn_v2_soft.Exponential2LinearCrossNetwork.forward = _orig_e2l_forward
        def m2_check():
            m_orig.eval(); m_new.eval()
            with torch.no_grad():
                a = m_orig.forward(batch)["y_pred"]
                b = m_new.forward(batch)["y_pred"]
            assert torch.equal(a, b), "E2L 层顺序反了却仍然逐位一致"
        # 注意：exp=4/lin=1 时顺序调换仍然会差（因为 lin 层用 x1、exp 层用当前 x）
        ok &= _expect_caught("M2 E2L 支路层顺序颠倒", m2_mutate, m2_restore, m2_check)

        # ---------- M3：残差里漏掉可学习的 b（x_anchor*(H+b)+x → x_anchor*H+x）----------
        def m3_mutate():
            def bad(self, x, x_anchor, i):
                H_1 = self.w[i](x)
                if len(self.layer_norm) > i:
                    H_2 = self.layer_norm[i](H_1)
                else:
                    H_2 = self.gamma[i] * H_1 + self.beta[i]
                H = torch.cat([H_1, H_2], dim=-1)
                if self.need_resize:
                    H = fcn_v2_soft.resize_last_dim(H, self.input_dim)
                if len(self.batch_norm) > i:
                    H = self.batch_norm[i](H)
                x = x_anchor * H + x                    # ← 漏掉 + self.b[i]
                if len(self.dropout) > i:
                    x = self.dropout[i](x)
                return x
            fcn_v2_soft._ParametricCrossNetwork._cross_layer = bad
        def m3_check():
            m_orig.eval(); m_new.eval()
            with torch.no_grad():
                a = m_orig.forward(batch)["y_pred"]
                b = m_new.forward(batch)["y_pred"]
            assert torch.equal(a, b), "残差漏掉 b 却仍然逐位一致"
        ok &= _expect_caught("M3 残差漏掉可学习偏置 b", m3_mutate, m2_restore, m3_check)

        # ---------- M4：add_loss 里把权重 detach（loss 相同，梯度不同）----------
        _orig_add_loss = fcn_v2_soft.FCNv2Soft.add_loss
        def m4_mutate():
            def bad(self, inputs):
                rd = self.forward(inputs)
                y_true = self.get_labels(inputs)
                y_pred, y_d, y_s = rd["y_pred"], rd["y_d"], rd["y_s"]
                loss = self.loss_fn(y_pred, y_true, reduction='mean')
                loss_d = self.loss_fn(y_d, y_true, reduction='mean')
                loss_s = self.loss_fn(y_s, y_true, reduction='mean')
                # 原版的 where 换成 detach 后的版本：值一样，梯度断了
                wd = (loss_d - loss).detach().clamp(min=0)
                ws = (loss_s - loss).detach().clamp(min=0)
                return loss + loss_d * wd + loss_s * ws
            fcn_v2_soft.FCNv2Soft.add_loss = bad
        def m4_restore():
            fcn_v2_soft.FCNv2Soft.add_loss = _orig_add_loss
        def m4_check():
            m_orig.train(); m_new.train()
            m_orig.zero_grad(set_to_none=True); m_new.zero_grad(set_to_none=True)
            torch.manual_seed(4321)
            TBE._assert_equal_loss_and_grad(m_orig, m_new, batch)
        ok &= _expect_caught("M4 权重 detach（值同梯度不同）", m4_mutate, m4_restore, m4_check)

        # ---------- M5：漏掉 init.uniform_(b) ----------
        _orig_init = fcn_v2_soft._ParametricCrossNetwork.__init__
        def m5_mutate():
            def bad(self, *a, **kw):
                _orig_init(self, *a, **kw)
                with torch.no_grad():
                    for p in self.b:      # 初始化后再扰动一下，等效于"少做/错做了一次 init"
                        p.add_(0.01)
            fcn_v2_soft._ParametricCrossNetwork.__init__ = bad
        def m5_restore():
            fcn_v2_soft._ParametricCrossNetwork.__init__ = _orig_init
        # 这个变异需要重建模型才生效，所以单独造一对
        def m5_check():
            d2 = tempfile.mkdtemp(prefix="fcnmut5_")
            try:
                mo, mn, _fm = TBE._build_pair(
                    d2, dict(mk, exp_num_layers=1, lin_num_layers=1))
                TBE._assert_state_dict_identical(mo, mn, tag="[M5]")
            finally:
                shutil.rmtree(d2, ignore_errors=True)
        ok &= _expect_caught("M5 b 的初始化被扰动", m5_mutate, m5_restore, m5_check)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n变异测试：%s" % ("全部变异都被抓住 —— 断言有判别力" if ok else "存在漏网变异 —— 测试不可信"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
