import geopandas as gpd
import math
def generate_network(road_shp, point_shp):
    # 读取道路shapefile
    roads = gpd.read_file(road_shp)
    from shapely.geometry import LineString

    edges = []
    nodes = []  # 存放唯一节点列表


    # FIXME 解决节点重复问题：在50m范围内寻找点，若发现则合并
    def euclidean_distance(point1, point2):
        """
        计算两个点之间的欧几里得距离
        point1, point2: (lat, lon)
        """
        lat1, lon1 = point1
        lat2, lon2 = point2
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


    def find_near_node(point, nodes, threshold=0.0005):
        """
        找到与 point 最近的节点，如果在阈值内则返回该节点，
        否则返回 None，调用处再把 point 加入 nodes。
        """
        for n in nodes:
            # 这里假设你的节点元组是 (lat, lon)
            # 如果是 (lon, lat)，记得对调
            distance = euclidean_distance(point, n)
            if distance <= threshold:
                return n
        return None  # <— 关键：没找到就返回 None


    for idx, row in roads.iterrows():
        line: LineString = row.geometry
        start_point = (line.coords[0][0], line.coords[0][1])
        end_point = (line.coords[-1][0], line.coords[-1][1])

        # 检查缓冲区内是否已有节点
        near_start = find_near_node(start_point, nodes)
        if near_start is None:
            nodes.append(start_point)
            near_start = start_point  # 自己就是新节点

        near_end = find_near_node(end_point, nodes)
        if near_end is None:
            nodes.append(end_point)
            near_end = end_point

        # 边: (起点, 终点, 属性)
        edges.append((near_start, near_end, row))

    import networkx as nx

    G = nx.Graph()  # 无向图
    for idx, node in enumerate(nodes, 1):
        G.add_node(node, Id=idx)  # 为每个节点添加Id属性，Id从1开始
    # 加入边，带属性（比如路段吸引力）
    for edge in edges:
        start, end, attributes = edge
        G.add_edge(start, end, **attributes)
    for u, v, data in G.edges(data=True):
        print(f"{u} -> {v},编号: {data['Id']}, 吸引力: {data['attraction']}")
    # ===============================
    # 2. 加载点 shp 文件
    # ===============================
    points_gdf = gpd.read_file(point_shp)

    # 这里假设 shp 文件中的 geometry 类型为 Point，
    # 我们将所有点的坐标（以 (x,y) 元组形式）存入一个集合，
    # 便于后续快速查找
    point_coords = set()
    for idx, row in points_gdf.iterrows():
        point = row.geometry
        # 注意：此处可以根据需要增加误差容忍度判断，本例采用精确匹配
        point_coords.add((point.x, point.y))
    # ===============================
    # 3. 对比网络中节点和点 shp 中的点,为网络中节点赋予属性值
    # ===============================
    from shapely.geometry import Point

    # 给点做缓冲区，比如 50 米（单位跟 CRS 有关）
    buffered_points = points_gdf.copy()
    print(buffered_points)
    buffered_points['geometry'] = buffered_points.geometry.buffer(50 / 111000)


    # 判断函数，node 是 (lon, lat)，注意坐标顺序
    def is_near_point_shapely(node, buffered_points):
        point_geom = Point(node)
        for idx, row in buffered_points.iterrows():
            if point_geom.within(row.geometry):
                print(row)
                return row['cid']  # 返回 point shp 中的 Id
        return None  # 没有命中


    # 遍历图中的节点，添加属性
    for node in G.nodes():
        cid = is_near_point_shapely(node, buffered_points)
        if cid is not None:
            G.nodes[node]['chushengdian'] = True
            G.nodes[node]['cid'] = cid
        else:
            G.nodes[node]['chushengdian'] = False
            G.nodes[node]['cid'] = None

    # 输出检查结果
    for node, data in G.nodes(data=True):
        print(f"节点 {node}， chushengdian: {data['chushengdian']}")
    return G

# G = generate_network("./含有引力值的数据/含有引力值的数据/road2.shp","./含有引力值的数据/含有引力值的数据/point.shp")
# starts = [n for n,d in G.nodes(data=True) if d.get("chushengdian")]
