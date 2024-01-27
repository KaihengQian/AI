from queue import PriorityQueue
import numpy as np


# 节点定义
class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent
        self.g = 0  # 从起始节点到当前节点的实际路径代价
        self.h = 0  # 从当前节点到目标节点的启发式估计代价
        self.f = 0  # f = g + h

    def __lt__(self, other):
        return self.f < other.f


# 使用曼哈顿距离的启发式函数
def manhattan_dist_heuristic(node1, node2):
    state1_indices = []
    state2_indices = []
    for value in range(8):
        indices1 = np.where(node1.state == value + 1)
        state1_indices.append(indices1[0])
        state1_indices.append(indices1[1])
        indices2 = np.where(node2.state == value + 1)
        state2_indices.append(indices2[0])
        state2_indices.append(indices2[1])

    state1_indices = np.array(state1_indices)
    state2_indices = np.array(state2_indices)

    manhattan_dist = np.sum(np.abs(state1_indices - state2_indices))
    return manhattan_dist


# 使用不重合度的启发式函数
def misalignment_heuristic(node1, node2):
    misalignment = np.sum(node1.state != node2.state)
    return misalignment


# A*算法实现
def a_star(start, end):
    # 定义移动方向（上、下、左、右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 存放未探索的节点
    open_list = PriorityQueue()
    # 存放已探索的节点
    closed_list = set()

    # 起始节点和目标节点
    start_node = Node(start)
    end_node = Node(end)

    # 从起始节点开始探索
    open_list.put((0, start_node))

    while not open_list.empty():
        # 将未探索的拥有最小f代价的节点作为当前节点
        _, current_node = open_list.get()
        closed_list.add(tuple(list(current_node.state.flat)))

        # 如果已经到达目标节点
        if np.array_equal(current_node.state, end_node.state):
            # 返回路径
            path = []
            while current_node is not None:
                path.append(list(current_node.state.flat))
                current_node = current_node.parent
            return path[::-1]

        # 当前空块所在位置
        current_zero_position = np.where(current_node.state == 0)

        # 否则，继续探索，即移动空块
        for direction in directions:
            # 移动后空块所在位置
            new_zero_position = (current_zero_position[0] + direction[0], current_zero_position[1] + direction[1])

            # 检查移动后空块所在位置
            if new_zero_position[0] < 0 or new_zero_position[0] > 2 or new_zero_position[1] < 0 or \
                    new_zero_position[1] > 2:
                continue

            # 移动后的新状态
            new_state = np.copy(current_node.state)
            (new_state[current_zero_position[0], current_zero_position[1]],
             new_state[new_zero_position[0], new_zero_position[1]]) = (
                new_state[new_zero_position[0], new_zero_position[1]],
                new_state[current_zero_position[0], current_zero_position[1]])

            # 检查移动后的新状态
            if tuple(list(new_state.flat)) in closed_list:
                continue

            # 如果符合要求，则以移动后的新状态创建新节点
            new_node = Node(new_state, current_node)
            new_node.g = current_node.g + 1
            new_node.h = manhattan_dist_heuristic(new_node, end_node)
            # new_node.h = misalignment_heuristic(new_node, end_node)
            new_node.f = new_node.g + new_node.h

            # 如果未探索节点队列中存在状态相同且f代价更低的节点，舍弃新节点
            pq = PriorityQueue()
            pq.queue = list(open_list.queue)
            while not pq.empty():
                _, node = pq.get()
                if np.array_equal(new_node.state, node.state) and new_node.f >= node.f:
                    break
            else:
                open_list.put((new_node.f, new_node))

    return None  # 如果没有找到路线，返回空
