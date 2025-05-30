import numpy as np
from skopt import Optimizer
import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import cdist
import re
# ============================================
# 请实现你的 R2 计算函数：
# 输入 a1, a2, ..., aP（满足 sum(a)=1），输出对应的 R2 值。
# 例如：
# def get_r(*a):
#     # 根据权重 a 计算 y_pred，再与真实 y 计算 R2
#     return r2_value
# ============================================
index1 = 0
def get_r(distance_matrix):
    """
    计算 R2 的占位函数。
    参数: a1, a2, ..., aP ————>接入shp赋吸引力模块
    返回: R2 值 ————>接入求R2模块
    """

    # 读取 shp 文件
    gdf = gpd.read_file("../实验数据/road.shp")
    # 2. 列名列表（顺序必须和权重列表对应）
    cols = [
        '影像绿', '口袋公', '容积率', '建筑密', 'richness_n', 'POI_shanno',
        'POI密度', '可达性', '绿视率', '天空开', '车行空', '人行空',
            '建筑物', '界面围', '安全设', '小径密', '地铁可', 'integratio', '选择度'
    ]
    # 修改 Id=2 的行的 "影像绿" 列为新的值，例如 0.123
    # 需要修改的 Id 列表
    # target_ids = [53, 54, 10, 9, 41, 8, 5]
    target_ids = [11, 54, 10, 57, 12, 13, 48,11,42]

    # 对指定 Id 的 POI_shanno 列值减去 0.2
    gdf.loc[gdf['Id'].isin(target_ids), '可达性'] += 0.6    # 2. 读取 CSV 文件（或你已提供的 DataFrame）
    weights_df = pd.read_csv("./数据/变量重要性结果(1).csv",encoding='gbk')  # 或你用 DataFrame 构造它
    # # 1. 读取文件
    # with open("../结果/权重/352.txt", "r", encoding="utf-8") as f:
    #     content = f.read()
    #
    # # 2. 提取合法的浮点数（支持科学计数法）
    # float_strs = re.findall(r'np\.float64\(([\deE\.\+-]+)\)', content)
    #
    # # 3. 转为 np.float64 类型并组成元组
    # float_values = list(map(np.float64, float_strs))
    # weights = tuple(float_values)
    #
    # # 4. 计算加权和并写入新字段
    # #    使用 zip 一次性完成点乘并累加
    # gdf['attraction'] = sum(w * gdf[c] for w, c in zip(weights, cols))
    # 3. 创建 name: 权重 的字典
    weight_map = dict(zip(weights_df['name'], weights_df['重要性']))

    # 4. 初始化新列
    gdf['attraction'] = 0

    # 5. 对每一列执行加权求和
    for col, weight in weight_map.items():
        if col in gdf.columns:
            gdf['attraction'] += gdf[col] * weight
        # print("----------------------------------")
        # print(gdf['attraction'])
        # print("----------------------------------")
    # 5. 删除原始指标列
    gdf = gdf.drop(columns=cols)
    # 6. 保存结果（可覆盖原文件，也可以输出新文件）
    save_name = "../实验数据/road2.shp"
    gdf.to_file(save_name)

    # -------------------------------------------------------
    # 仿真模拟
    # import sys
    # sys.path.append('../')
    from 行人仿真模块_多行人仿真并行 import zhixingfangzhen
    # print("参数权重：",a)#a[0]
    global index1
    index1 = index1 + 1
    r = zhixingfangzhen(index1,distance_matrix,road_shp=save_name, point_shp="../实验数据/出生点.shp",speed=1.0,max_time=1800,sumPeople=260)

    return r

def get_jvli():
    # === 1. 读取两个点状shp文件 ===
    gdf1 = gpd.read_file("../实验数据/建筑点.shp")  # 假设有 N 个点
    gdf2 = gpd.read_file("../实验数据/出生点.shp")  # 假设有 M 个点

    # === 2. 确保坐标系一致，并转为米制（如 EPSG:3857） ===
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    if gdf1.crs.is_geographic:
        gdf1 = gdf1.to_crs(epsg=3857)
        gdf2 = gdf2.to_crs(epsg=3857)

    # === 3. 提取坐标和 ID ===
    coords1 = np.array([[geom.x, geom.y] for geom in gdf1.geometry])
    coords2 = np.array([[geom.x, geom.y] for geom in gdf2.geometry])

    ids1 = gdf1['Id'].astype(str).values
    ids2 = gdf2['cid'].astype(str).values

    # === 4. 计算距离矩阵 ===
    distance_array = cdist(coords1, coords2, metric='euclidean')  # shape: [len(gdf1), len(gdf2)]

    # === 5. 构建带有 Id 和 cid 的 DataFrame ===
    distance_matrix = pd.DataFrame(distance_array, index=ids1, columns=ids2)

    # === 6. 显示示例结果 ===
    print("带有 Id 和 cid 的距离矩阵（单位：米）:")
    print(distance_matrix)
    return distance_matrix
if __name__ == "__main__":
    distance_matrix = get_jvli()

    # TODO 调整参数优化 权重个数
    P = 19
    # TODO 调整参数n_iter：迭代次数
    # best_w, best_r2 = optimize_weights(P,distance_matrix, n_iter=2, random_state=42)
    get_r(distance_matrix)
    # print("\n=== 最终最优参数 ===")
    # print(f"weights = {best_w.round(6).tolist()}\nR2 = {best_r2:.6f}")
