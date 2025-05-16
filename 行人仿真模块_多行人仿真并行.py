import networkx as nx
import gym
from gym import spaces
import random
import sys
sys.path.append('./粘菌算法/SMA/Python')
from my_sma import main as sma_main
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.metrics import r2_score
from typing import Optional, Tuple
import pandas as pd
# ─── 1. 假设已经有一个 NetworkX 图 G ─────────────────────────


# ─── 2. 定义 Gym 环境 ────────────────────────────────
class PedestrianEnv(gym.Env):
    """
    Gym 环境：行人在路网中行走，每步根据吸引力概率选路，
    且不会回头走上一步刚走过的路。
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, G,distance_matrix,  max_time, speed=1.0,max_deg=4,sumPeople=30):#所有“没有默认值的参数”必须出现在“有默认值的参数”前面。
        super().__init__()#调用父类（基类）的初始化方法。
        self.distance_matrix = distance_matrix
        self.sumPeople = sumPeople
        self.G = G
        self.speed = speed
        self.max_time = max_time
        self.max_deg = max_deg

        # 动作空间：0~max_deg-1
        self.action_space = spaces.Discrete(self.max_deg)
        # 观测：剩余时间 + max_deg 条边的吸引力
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1 + self.max_deg,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        # 加载 CSV 文件
        df = pd.read_csv("./实验数据/建筑物点_出生点.csv", encoding='utf-8')

        # 加载距离矩阵（这里假设你已经从 shapefile 计算好了）
        # 距离矩阵的行是 Id（字符串），列是 cid（字符串）
        distance_matrix = self.distance_matrix

        # 从所有 Id 中，按概率列随机抽取一个 Id（只取一个）
        target_id = df[['Id', '概率']].drop_duplicates().sample(
            weights='概率',
            n=1
        )['Id'].values[0]

        # 2. 找出该 Id 对应的所有 cid（保持为字符串用于索引）
        target_cids = df[df['Id'] == target_id]['cid'].astype(str).tolist()

        # 3. 提取该 Id 到这些 cid 的距离
        # 确保 Id 和 cid 都是字符串类型（用于索引）
        id_str = str(target_id)
        distances = distance_matrix.loc[id_str, target_cids]

        # 4. 将距离转换为概率（距离越近，概率越高），做反比例归一化（防止除 0）
        eps = 1e-6
        inv_dist = 1 / (distances + eps)
        probs = inv_dist / inv_dist.sum()

        # 5. 输出结果
        result = pd.DataFrame({'cid': target_cids, 'probability': probs.values})
        # 按概率随机选择一个 cid
        selected_cid = result.sample(weights=result['probability'], n=1)['cid'].values[0]

        print(f"最终选择的 cid 是: {selected_cid}")
        for n, data in self.G.nodes(data=True):
            print(f"节点编号: {n}, cid: {data.get('cid')}")
        for n, d in self.G.nodes(data=True):
            if str(d.get('cid')) == str(selected_cid):
                self.current = n
                break
        else:
            raise ValueError(f"找不到 cid={selected_cid} 对应的图节点！")
        self.prev_node = None            # 上一步走过的节点，重置为空
        self.time_left = self.max_time
        self.path = [self.current]
        return self._get_obs()

    def _get_obs(self):
        # 当前可选邻居，剔除上一步节点
        nbrs = [n for n in self.G.neighbors(self.current)
                if n != self.prev_node]
        # 吸引力列表
        atts = [self.G.edges[self.current, n]["attraction"] for n in nbrs]
        # pad to max_deg
        atts = (atts + [0.0]*self.max_deg)[:self.max_deg]
        return np.array([self.time_left] + atts, dtype=np.float32)

    def step(self, action):
        # 构造候选邻居列表（不包括 prev_node）
        nbrs = [n for n in self.G.neighbors(self.current)
                if n != self.prev_node]
        # 非法动作：超出 nbrs 长度
        if action >= len(nbrs):
            return self._get_obs(), -10.0, True, {}

        # 执行动作
        next_node = nbrs[action]
        edge = self.G.edges[self.current, next_node]
        time_cost = edge["long"] / self.speed

        # 更新状态
        self.time_left -= time_cost
        self.prev_node = self.current
        self.current = next_node
        self.path.append(next_node)

        reward = edge.get("attraction", 0.0)
        done   = (self.time_left <= 0)
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"At {self.current}, time left {self.time_left:.1f}, path: {self.path}")

    def close(self):
        pass


def sma_policy(obs, env: PedestrianEnv, max_iter=100): #max_iter：最大迭代次数
    """
    用粘叶菌算法计算当前邻边的选择概率并采样。
    obs: 当前观测（这里只是占位，用不到）
    env: 环境实例，里面保存了图 G、sumPeople、current、prev_node
    max_iter: SMA 算法的迭代次数
    """
    # 1. 列出所有可选的邻居（不回头）
    nbrs = [n for n in env.G.neighbors(env.current)
            if n != getattr(env, "prev_node", None)]
    if not nbrs:
        return 0

    # 2. 构造吸引力数组
    attractions = np.array([env.G.edges[env.current, n]["attraction"]
                            for n in nbrs], dtype=np.float32)

    # 3. 调用 SMA 主函数，得到选择概率分布
    #    sma_main 返回一个和 attractions 等长的概率向量
    probs = sma_main(attractions, env.sumPeople, max_iter)

    # 4. 按概率采样
    idx = np.random.choice(len(nbrs), p=probs)
    return idx



# ─── 4. 模拟 100 个行人独立决策 ─────────────────
import matplotlib.pyplot as plt


def zhixingfangzhen(index1,distance_matrix,road_shp, point_shp,speed,max_time,sumPeople):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    from 道路建模 import generate_network

    # TODO 设置参数，点和线shp的位置
    G = generate_network(road_shp,
                         point_shp)

    # 单个行人仿真函数
    def simulate_one_person(speed, max_time, max_deg):

        env = PedestrianEnv(G,distance_matrix, speed=speed, max_time=max_time, max_deg=max_deg)

        obs = env.reset()
        done = False
        prev = env.current
        path = []

        while not done:
            a = sma_policy(obs, env, max_iter=100)
            obs, _, done, _ = env.step(a)
            curr = env.current
            path.append((prev, curr))
            prev = curr

        return path


    # 并行运行多个行人仿真
    all_paths = Parallel(n_jobs=10)(  # 根据你的 CPU 调整 n_jobs
        delayed(simulate_one_person)(speed, max_time, 4) for _ in range(sumPeople) #并行执行多次 simulate_one_person(speed, max_time, 4)
    )
    print("路径总表：",all_paths)
    # 汇总每条边的通行次数
    edge_count = defaultdict(int)
    for path in all_paths:
        for u, v in path:
            e = tuple(sorted((u, v)))  # 确保无向边一致性
            edge_count[e] += 1

    # 写回 count 属性
    for (u, v), cnt in edge_count.items():
        G.edges[u, v]["count"] = cnt
    import numpy as np
    # 提取数据
    values = [(data['count'], data['people_v'])
              for _, _, data in G.edges(data=True)
              if 'count' in data and 'people_v' in data]
    #('A', 'B', {'count': 12, 'people_v': 15})
    #('C', 'D', {'count': 8, 'people_v': 6})
    # ('E', 'F', {'count': 0, 'people_v': 0})
    #value = [(12, 15), (8, 9), (0, 0), ...]

    if values:
        y_pred, y_true = zip(*values)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)

        # 将预测值和真实值归一化为比例分布
        y_pred_ratio = y_pred / y_pred.sum() if y_pred.sum() != 0 else y_pred
        y_true_ratio = y_true / y_true.sum() if y_true.sum() != 0 else y_true
        print("路径仿真人数和真实人数：", values)
        # 计算归一化后的 R²
        r2 = r2_score(y_true_ratio, y_pred_ratio)
        print(f"归一化分布 R² = {r2:.4f}")

        print("绘制热力图")
        pos = {n: n for n in G.nodes()}  # 假设节点标签就是坐标或名字

        # 提取每条边的 count 值
        counts = np.array([G.edges[u, v].get("count", 0) for u, v in G.edges()])
        # 构建线宽和颜色映射
        # 线宽：最少0.5，最多4
        widths = 0.5 + 3 * (counts / counts.max())
        # 颜色：使用 inferno colormap
        cmap = plt.cm.inferno
        # 归一化 counts 到 [0,1]
        norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
        colors = cmap(norm(counts))

        plt.figure(figsize=(8, 6))
        # 先画节点
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color="lightgray")
        # 画带热力的边
        nx.draw_networkx_edges(
            G, pos,
            edgelist=list(G.edges()),
            width=widths,
            edge_color=colors,
            edge_cmap=cmap,
            edge_vmin=counts.min(),
            edge_vmax=counts.max()
        )
        # 添加 colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="被路过次数")

        # 可选：在节点上标注名称
        # nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title(f"道路热力图 —— 行人路过次数  (R2:{r2:.4f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./结果/图片/{index1}.png")
        plt.show()
        return r2
    else:
        print("没有有效数据。")
        return "没有有效数据。"
    # —— 2. 绘制热力图 —— #

