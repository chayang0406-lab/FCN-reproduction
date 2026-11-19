"""反解作者旧版日志的参数量公式，确认 16,138,194 的来源。

已确认事实：
  embedding 参数 = sum(vocab_size) x embedding_dim = 910,747 x 16 = 14,571,952
  作者主日志总参数 = 16,138,194
  => 两个交叉网络合计 = 16,138,194 - 14,571,952 = 1,566,242
  => 单个网络 = 783,121
  => 减去 fc 层 (Linear(624,1) -> 624 权重 + 1 偏置 = 625) = 782,496
  => 旧日志 exp=3/lin=3 共 6 层 => 每层 = 130,416

那么 130,416 是什么？本脚本枚举可能的层内结构。
"""
D = 39 * 16          # input_dim = 624
EMB = 910747 * 16
TOTAL = 16138194
N_LAYERS = 6         # ECN_num_layers=3 + LCN_num_layers=3
N_NETS = 2

net_total = TOTAL - EMB
per_net = net_total / N_NETS
per_layer = (per_net - (D + 1)) / N_LAYERS

print("=== 已知量 ===")
print("  D (input_dim)        =", D)
print("  embedding params     =", EMB)
print("  交叉网络合计         =", net_total)
print("  单网络               =", per_net)
print("  单层                 =", per_layer)
print()

print("=== 反解每层结构 ===")
print("  HEAD 版每层 = w(D x D//2) + gamma(D//2) + beta(D//2) + b(D)")
print("              = %d x %d + %d + %d + %d = %d" % (D, D//2, D//2, D//2, D, D*(D//2)+D//2+D//2+D))
print()
# 猜：w = D x (D//3), 再加 b 向量 D
cand = D * (D // 3) + D
print("  猜法A: w(D x D//3) + b(D) = %d x %d + %d = %d" % (D, D//3, D, cand))
print("        与实测单层 %.0f 是否相等: %s" % (per_layer, cand == per_layer))
print()

print("=== 用猜法A 完整验算 ===")
per_layer_A = D * (D // 3) + D          # 130416
per_net_A = N_LAYERS * per_layer_A + (D + 1)
total_A = EMB + N_NETS * per_net_A
print("  单层     = %d" % per_layer_A)
print("  单网络   = %d 层 x %d + fc(%d) = %d" % (N_LAYERS, per_layer_A, D + 1, per_net_A))
print("  总参数   = %d + 2 x %d = %d" % (EMB, per_net_A, total_A))
print("  作者日志 = %d" % TOTAL)
print("  残差     = %d" % (total_A - TOTAL))
print()
if total_A == TOTAL:
    print("  ==> 完全吻合，残差为 0。")
    print("      旧版每层是 w(D x D//3, 无 bias) + 一个 D 维向量 b；")
    print("      HEAD 版每层是 w(D x D//2, 无 bias) + gamma(D//2) + beta(D//2) + b(D)。")
    print("      两者输出宽度不同（D//3=208 vs D//2=312），所以参数量对不上是架构差异，")
    print("      不是我们配错了。")
else:
    print("  ==> 不吻合，说明猜法A 仍然不对，残差需继续追。")

