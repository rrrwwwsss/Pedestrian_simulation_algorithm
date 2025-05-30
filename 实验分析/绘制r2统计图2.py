import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns

# 生成极坐标螺旋
n_points = 1000
theta = np.linspace(0, 8 * np.pi, n_points)  # 角度，越多圈越螺旋
r = np.linspace(0.2, 1.0, n_points)          # 半径递增

# 转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 构造螺旋上的值并映射到 -0.2 到 0.6
z = np.linspace(-0.2, 0.6, n_points) + 0.05 * np.sin(10 * theta)  # 加一点扰动增加拟合变化

# 绘图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=z, label="Spiral Scatter")
sns.regplot(x=x, y=z, scatter=False, color='red', label="Trend Line", lowess=True)
plt.xlabel("X")
plt.ylabel("Value (z)")
plt.title("Spiral Upward Trend")
plt.legend()
plt.grid(True)
plt.show()