from A_star_algorithm import *


if __name__ == '__main__':
    N, M, K = map(int, input("请输入：\n").split())
    downhill_roads = []
    for _ in range(M):
        X, Y, D = map(int, input().split())
        downhill_roads.append([X, Y, D])

    # 初始位置
    orig_position = 1
    # 目标位置
    goal_position = N

    dists, paths = a_star(orig_position, goal_position, downhill_roads, K, N)
    for dist in dists:
        print(dist)

    for path in paths:
        print(path)
