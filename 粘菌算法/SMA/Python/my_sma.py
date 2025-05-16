import numpy as np

from select_Optimize_algorithm import SMA

def road_choice_objf(attractions):
    def objf(x):
        """
        目标函数
        x: 当前解(各道路的选择权重)
        返回: 与道路吸引力的匹配程度(越小越好)
        """
        # 添加一个小的常数避免除以零
        epsilon = 1e-10
        # 归一化x使其和为1
        probs = x / (np.sum(x) + epsilon)
        # 计算与原始吸引力的差异
        error = np.sum((probs - attractions/np.sum(attractions))**2)
        return error  
    return objf

def main(roads,population_size,maximum_iterations):
    # 使用示例:
    n_roads = len(roads)
    # 原始道路吸引力
    attractions = np.array(roads)
    # 构造目标函数
    objf = road_choice_objf(attractions)
    # 运行SMA算法
    # 这里是 3，代表有 3 条道路，每条道路的选择概率是一个变量 ;30 种群数量，也就是一共有 30 个黏菌个体在解空间里爬;
    # 100最大迭代次数，也就是最多优化 100 轮
    # [0] * n_roads:生成一个长度为n_roads的列表，每个元素都是0，代表初始化每个道路的选择概率都是0
    # 把这个目标函数传入 SMA 后，SMA 会返回一个最优解，表示在各道路选择权重上的最佳分布，
    # 通过归一化可以得到与道路原始吸引力分布吻合的选择概率。
    result = SMA(objf, [0]*n_roads, [1]*n_roads, n_roads, population_size, maximum_iterations)

    # 获取结果并归一化为概率
    probs = result.bestIndividual / np.sum(result.bestIndividual)
    print("道路吸引力:", attractions)
    print("选择概率:", probs)
    return probs