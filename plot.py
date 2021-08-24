"""
@Author:        禹棋赢
@StartTime:     2021/8/24 9:19
@Filename:      plot.py
"""
import matplotlib.pyplot as plt
# from d2l import torch as d2l
from matplotlib.ticker import MultipleLocator

x = [
    27.17, 45.5, 53.27, 58.04, 65.65, 67.42, 72.17, 74.11, 78.63, 80.01,
    81.75, 84.27, 85.82, 87.41, 89.43, 90.06, 90.79, 92.95, 92.26, 93.52,
    77.32,
]

y = [
    23.46, 35.97, 40.40, 43.25, 48.11, 49.24, 52.83, 51.33, 58.00, 60.05,
    60.93, 59.34, 62.35, 61.97, 62.86, 62.90, 63.16, 64.49, 62.54, 61.68,
    63.04,
]

with open("logs/pgd.log", "r") as f:
    lines = f.readlines()

fgt, util = [0], [0]

for li in lines:
    if li.startswith("forget"):
        a = li.split()
        fgt.append(float(a[3][:-1]))
        util.append(float(a[7]))

plt.plot(fgt, label="ep1 acc")
plt.plot(util, label="util rate")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()

xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
ax = plt.gca()
ax.xaxis.set_minor_locator(xminorLocator)

plt.grid(which="both", linestyle="--")
plt.show()
