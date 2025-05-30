import numpy as np
from skopt import Optimizer
import geopandas as gpd
import sys
import pandas as pd
from scipy.spatial.distance import cdist
# ============================================
# 请实现你的 R2 计算函数：
# 输入 a1, a2, ..., aP（满足 sum(a)=1），输出对应的 R2 值。
# 例如：
# def get_r(*a):
#     # 根据权重 a 计算 y_pred，再与真实 y 计算 R2
#     return r2_value
# ============================================
index1 = 0
def get_r(*a,distance_matrix):
    """
    计算 R2 的占位函数。
    参数: a1, a2, ..., aP ————>接入shp赋吸引力模块
    返回: R2 值 ————>接入求R2模块
    """

    # 读取 shp 文件
    gdf = gpd.read_file("./实验数据/road.shp")
    # 2. 列名列表（顺序必须和权重列表对应）
    cols = [
        '影像绿', '口袋公', '容积率', '建筑密', 'richness_n', 'POI_shanno',
        'POI密度', '可达性', '绿视率', '天空开', '车行空', '人行空',
        '建筑物', '界面围', '安全设', '小径密', '地铁可', 'integratio', '选择度'
    ]
    # cols = []
    # for i in range(1, 20):
    #     cols.append(f"X{i}")
    # 3. 你的权重列表（长度要跟 cols 一致）
    weights = a  # 替换成你的具体数值

    # 4. 计算加权和并写入新字段
    #    使用 zip 一次性完成点乘并累加
    gdf['attraction'] = sum(w * gdf[c] for w, c in zip(weights, cols))
    # 5. 删除原始指标列
    gdf = gdf.drop(columns=cols)
    # 6. 保存结果（可覆盖原文件，也可以输出新文件）
    save_name = "./实验数据/road2.shp"
    gdf.to_file(save_name)

    # -------------------------------------------------------
    # 仿真模拟
    from 行人仿真模块_多行人仿真并行 import zhixingfangzhen
    # print("参数权重：",a)#a[0]
    global index1
    index1 = index1 + 1
    with open(f"./结果/权重/{index1}.txt","w") as f:
        f.write(str(weights))
    print(f"已保存到./结果/权重/{index1}.txt")
    r = zhixingfangzhen(index1,distance_matrix,road_shp=save_name, point_shp="./实验数据/出生点.shp",speed=1.0,max_time=1800,sumPeople=260)

    return r


def optimize_weights(P, distance_matrix,n_iter=50, random_state=42):
    """
    在 P 个非负权重且和为 1 的约束下，
    利用贝叶斯优化不断迭代寻找使 R2 最大化的权重组合。

    参数:
        P (int): 权重个数
        n_iter (int): 迭代次数
        random_state (int): 随机种子，保证可复现
    返回:
        最优权重列表和最佳 R2
    """
    # 定义无约束空间：P-1 个 u 变量
    space = [(-5.0, 5.0)] * (P - 1)
    opt = Optimizer(space, random_state=random_state)

    for i in range(1, n_iter + 1):
        # Ask: 在 (u1,...,u_{P-1}) 空间采样
        u_vec = opt.ask()

        # 指数映射到单纯形：a_i = exp(u_i) / sum(exp(u) + 1)
        exps = np.exp(u_vec)
        exps = np.concatenate([exps, [1.0]])  # 最后一维固定为 1
        weights = exps / np.sum(exps)

        # 计算 R2
        r2 = get_r(*weights, distance_matrix=distance_matrix)

        # Tell: skopt 最小化目标，传入 1-R2
        opt.tell(u_vec, 1 - r2)

        print(f"Iter {i:03d}: weights={weights.round(4).tolist()} R2={r2:.6f}")

    # 获取最优结果
    best_idx = int(np.argmin(opt.yi))
    best_u = opt.Xi[best_idx]
    exps = np.exp(best_u)
    exps = np.concatenate([exps, [1.0]])
    best_weights = exps / np.sum(exps)
    best_r2 = 1 - opt.yi[best_idx]

    return best_weights, best_r2

def get_jvli():
    # === 1. 读取两个点状shp文件 ===
    gdf1 = gpd.read_file("./实验数据/建筑点.shp")  # 假设有 N 个点
    gdf2 = gpd.read_file("./实验数据/出生点.shp")  # 假设有 M 个点

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
    best_w, best_r2 = optimize_weights(P,distance_matrix, n_iter=2, random_state=42)
    print("\n=== 最终最优参数 ===")
    print(f"weights = {best_w.round(6).tolist()}\nR2 = {best_r2:.6f}")
