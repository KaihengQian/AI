from queue import PriorityQueue


# 节点定义
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # 从起始节点到当前节点的实际路径代价
        self.h = 0  # 从当前节点到目标节点的启发式估计代价
        self.f = 0  # f = g + h

    def __lt__(self, other):
        return self.f < other.f


# 使用Dijkstra算法计算启发式函数
def dijkstra_heuristic(node1, node2, roads, N):
    inf = 1e6
    dist = [inf] * (N + 1)
    dist[node1.position] = 0
    open_list = PriorityQueue()
    open_list.put((0, node1.position))

    while not open_list.empty():
        current_dist, current_position = open_list.get()

        if current_position == node2.position:
            return dist[node2.position]

        if current_dist > dist[current_position]:
            continue

        for road in roads:
            if road[0] == current_position:
                new_position = road[1]
                cost = road[2]
                if dist[current_position] + cost < dist[new_position]:
                    dist[new_position] = dist[current_position] + cost
                    open_list.put((dist[new_position], new_position))

    return dist[node2.position]


# A*算法实现
def a_star(start, end, roads, k, N):
    # 存放未探索的节点
    open_list = PriorityQueue()
    # 存放已探索的节点
    closed_list = set()

    # 起始节点和目标节点
    start_node = Node(start)
    end_node = Node(end)

    # 从起始节点开始探索
    open_list.put((0, start_node))

    # 存放已找到路径的代价和具体信息
    dists = []
    paths = []

    while not open_list.empty():
        # 将未探索的拥有最小f代价的节点作为当前节点
        _, current_node = open_list.get()
        closed_list.add(current_node.position)

        # 如果已经到达目标节点
        if current_node.position == end_node.position:
            # 路径代价
            dist = current_node.g
            dists.append(dist)

            # 路径具体信息
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            paths.append(path[::-1])

            # 如果已经找到k条路径
            if len(dists) == k:
                break

            # 否则，继续探索
            closed_list.clear()
            continue

        # 否则，继续探索
        for road in roads:
            # 只选择下坡路径
            if road[0] == current_node.position and road[1] > road[0]:
                new_position = road[1]

                # 如果探索到的位置不符合要求，舍弃
                if new_position in closed_list:
                    continue

                # 否则，以探索到的位置创建新节点
                new_node = Node(new_position, current_node)
                new_node.g = current_node.g + road[2]
                new_node.h = dijkstra_heuristic(new_node, end_node, roads, N)
                new_node.f = new_node.g + new_node.h

                # 由于需要寻找k条较短路径，所以如果未探索节点队列中存在位置相同且f代价更低的节点，也不舍弃新节点
                open_list.put((new_node.f, new_node))

    if len(dists) < k:
        for _ in range(k-len(dists)):
            dists.append(-1)

    return dists, paths
