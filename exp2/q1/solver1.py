from A_star_algorithm import *


if __name__ == '__main__':
    # 初始状态
    tmp_input = input("请输入初始状态：\n")
    tmp_input = [int(digit) for digit in str(tmp_input)]
    orig_state = [tmp_input[i:i+3] for i in range(0, len(tmp_input), 3)]
    # 目标状态
    goal_state = [[1, 3, 5], [7, 0, 2], [6, 8, 4]]

    path = a_star(orig_state, goal_state)
    print(len(path)-1)

    for state in path:
        print(state)
