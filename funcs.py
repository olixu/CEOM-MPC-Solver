# -*- encoding: utf-8 -*-
"""
@File    :   funcs.py
@Time    :   2022/11/26 09:58:03
@Author  :   olixu 
@Version :   1.0
@Contact :   273601727@qq.com
@WebSite    :   https://blog.oliverxu.cn
"""

# here put the import lib
import os
import numpy as np
import symengine

action_dim = 2
state_dim = 2
N = 10


def generate_cost_func():
    """
    功能：生成优化问题，返回的是一个lambdify函数
    输入：
    1. 函数体内部参数修改，包括：
    a. 控制时域：N
    b. 状态空间的维度：action_dim
    c. 动作空间的维度：state_dim
    
    输出：
    1. 构建的数值表达式：cost_func
    """
    N = 10
    state_dim = 2
    action_dim = 2
    # 通过符号表达式定义目标函数
    action_list = [symengine.Symbol("a{}".format(i)) for i in range(action_dim * N)]
    initial_state_list = [symengine.Symbol("x{}".format(i)) for i in range(state_dim)]
    state1_list = []
    state2_list = []
    state1_list.append(initial_state_list[0])
    state2_list.append(initial_state_list[1])
    obj = 0

    for i in range(N):
        state1_list.append(
            -0.3 * state2_list[i] * symengine.cos(state1_list[i])
            + action_list[action_dim * i]
        )
        state2_list.append(
            1.01 * state2_list[i]
            + 0.2 * symengine.sin(state1_list[i] ** 2)
            + action_list[action_dim * i + 1]
        )
        obj += state1_list[i] ** 2 + state2_list[i] ** 2 + action_list[i] ** 2
    obj += state1_list[-1] ** 2 + state2_list[-1] ** 2

    cost_func = symengine.lambdify(
        action_list + initial_state_list, [obj], dtype=np.float64, backend="llvm"
    )
    return cost_func


def compute_loss(initial_state: np.array, action: np.array):
    """
    功能：给定一个初始状态，一串动作轨迹，计算状态轨迹和损失
    输入：
    1. 初始状态
    2. 动作轨迹
    
    输出：
    1. 状态轨迹
    2. 损失函数目标值
    """
    steps = N
    state_trajectory = initial_state[np.newaxis, :]
    for i in range(int(steps)):
        state1_temp = (
            -0.3 * state_trajectory[i][1] * np.cos(state_trajectory[i][0])
            + action[2 * i]
        )
        state2_temp = (
            1.01 * state_trajectory[i][1]
            + 0.2 * np.sin(state_trajectory[i][0] ** 2)
            + action[2 * i + 1]
        )
        state_trajectory = np.append(
            state_trajectory, np.array([[state1_temp, state2_temp]]), axis=0
        )
    return state_trajectory, np.sum(action ** 2) + np.sum(state_trajectory ** 2)
