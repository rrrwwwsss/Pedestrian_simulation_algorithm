import pandas as pd
import re

# 1. 建成环境因子名称列表
factors = [
    '影像绿', '口袋公', '容积率', '建筑密', 'richness_n', 'POI_shanno',
    'POI密度', '可达性', '绿视率', '天空开', '车行空', '人行空',
    '建筑物', '界面围', '安全设', '小径密', '地铁可', 'integratio', '选择度'
]
index1 = 608
# 2. 读取 txt 文件
with open(f"../结果/权重/{index1}.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 3. 提取 np.float64(...) 里的数字
float_values = re.findall(r'np\.float64\(([\deE\.\+-]+)\)', content)
float_values = list(map(float, float_values))

# 4. 构造 DataFrame，添加建成环境因子列
df = pd.DataFrame({
    'Index': range(len(float_values)),
    'Value': [round(v, 4) for v in float_values],
    '建成环境因子': factors[:len(float_values)]  # 保证不越界
})

# 5. 输出查看
print(df)
df.to_csv(f'./分析结果/{index1}.csv', index=False)
