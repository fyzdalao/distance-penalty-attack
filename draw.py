import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

attractor_interval = 6
alpha = 0.7
T = attractor_interval

x = np.linspace(0, 20, 2000)
attractor = (np.floor(x / attractor_interval) + 0.5) * attractor_interval

target_lnr = attractor - alpha * (x - attractor)
target_sin = x - alpha * attractor_interval * np.sin( ( 1 - 2 / attractor_interval * (x - attractor)) * np.pi )


l_top = x + 0.3 *  (T - x % T)
l_low = x + 0.3 * T - 1.2 * (x % T)


plt.plot(x, l_low, color='blue')
plt.plot(x, l_top, color='red')
plt.title('output loss')
plt.show()
#
#
# def target(x, T):
#     bias_x = x % T
#     zero = (x // T) * T
#
#     a = 1
#     b = 0
#
#     # bias_y = a * bias_x + b * T
#
#     if bias_x < 0.1 * T:
#         a = 6
#         b = 0
#     elif bias_x < 0.5 * T:
#         ratio = (bias_x - 0.1 * T) / (0.4 * T)
#         flag = 0
#         if ratio < 0.1: flag = 1
#         elif ratio > 0.2 and ratio < 0.3: flag = 1
#         elif ratio > 0.4 and ratio < 0.5: flag = 1
#         elif ratio > 0.6 and ratio < 0.7: flag = 1
#         elif ratio > 0.8 and ratio < 0.9: flag = 1
#
#         if flag == 0:
#             a = 0.25
#             b = 23/40
#         else:
#             a = -0.25
#             b = 5/8
#     elif bias_x < 0.9 * T:
#         ratio = (bias_x - 0.5 * T) / (0.4 * T)
#         flag = 0
#         if ratio < 0.1: flag = 1
#         elif ratio > 0.2 and ratio < 0.3: flag = 1
#         elif ratio > 0.4 and ratio < 0.5: flag = 1
#         elif ratio > 0.6 and ratio < 0.7: flag = 1
#         elif ratio > 0.8 and ratio < 0.9: flag = 1
#
#         if flag == 0:
#             a = 0.25
#             b = 11/40
#         else:
#             a = -0.25
#             b = 21/40
#     else:
#         a = 7
#         b = -6
#
#     return a * bias_x + b * T + zero
#
#
#
#
# xs = np.zeros(22000, dtype=float)
# ys = np.zeros(22000, dtype=float)
#
# it = 1
# i = 0
# while i < 21:
#     xs[it] = i
#     ys[it] = target(i, 6)
#     i += 0.003
#     it += 1
#
# plt.scatter(xs, ys, s=0.02)
# plt.show()


ans = 33.3 % 10.2
print(ans)






















# logit1 = torch.tensor([10.0,9.5,9.8])
#
# logit2 = torch.tensor([10.0,0.5,0.7])
#
# logit3 = torch.tensor([10.0,1.0,1.0])
#
# logit4 = torch.tensor([10.0,1.0,1.0])
#
# logits1 = torch.stack( (logit1, logit2) ,0 )
#
# logits2 = torch.stack( (logit3, logit4) ,0 )
#
# print(logits1)
# print(logits2)
#
# for i in range(1):
#     logits1 += logits2
#
# logits1 /= 2
#
# print(logits1)


