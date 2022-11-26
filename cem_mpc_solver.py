# -*- encoding: utf-8 -*-
"""
@File    :   cem_mpc_solver.py
@Time    :   2022/11/26 09:43:02
@Author  :   olixu 
@Version :   1.0
@Contact :   273601727@qq.com
@WebSite    :   https://blog.oliverxu.cn
"""

# here put the import lib
from multiprocessing import pool
from re import A
import symengine

import numpy as np
import scipy
import scipy.stats
import time
import multiprocessing
import sys
import pdb
import pickle
import os
from funcs import compute_loss, action_dim, N


def cem_optimize(
    cost_func,
    sample_N: int,
    Nel: int,
    initial_state: np.array,
    means: list,
    stds: list,
):
    """
    功能：CEOM实现

    输入：
    1. 符号表达式
    3. 采样个数：sample_N
    4. 选取最好的个数：Nel
    5. 初始状态
    6. initial_mean
    7. initial_std

    输出：
    1. mean_list也等于action_trajectory
    2. std_list
    3. state_tracejectory
    5. running_time
    """
    start_time = time.time()
    action_sampled = scipy.stats.norm(loc=means, scale=stds).rvs(
        size=(sample_N, action_dim * N)
    )
    obj_values = cost_func(
        np.concatenate((action_sampled, np.tile(initial_state, (sample_N, 1))), axis=1)
    )
    # 进行排序
    sort_index = np.argsort(obj_values)
    nel_sort_index = sort_index[0:Nel]
    means[:] = action_sampled[nel_sort_index].mean(axis=0)
    stds[:] = action_sampled[nel_sort_index].std(axis=0)
    state_trajectory, loss = compute_loss(initial_state, means)
    return state_trajectory, loss, time.time() - start_time


def qp_solver(cost_func, initial_state):
    """
    功能：使用CEOM算法求解一次二次型优化问题

    输入：
    1. cost_func: 目标函数
    3. initial_state: 初始状态

    输出：
    1. action: 
    2. state: 
    3. loss: 
    """
    # pdb.set_trace()
    pre_loss = 1.0
    # 定义参数
    sample_N = 2000
    Nel = 10
    means = np.array([0 for i in range(action_dim * N)], dtype=np.float64)
    stds = np.array([1 for i in range(action_dim * N)], dtype=np.float64)
    for i in range(100):
        state_trajectory, loss, runtime = cem_optimize(
            cost_func,
            sample_N=sample_N,
            Nel=10,
            initial_state=initial_state,
            means=means,
            stds=stds,
        )
        if np.abs(pre_loss - loss) < 1e-8:
            # print("损失是：", loss)
            break
        pre_loss = loss
    return means[0:2], state_trajectory[1], loss


def mpc_solver(cost_func, initial_state, trajectory, log=True):
    """
    功能：使用qp_solver来求解MPC问题：一个MPC问题相当于使用qp_solver求解N次数，得到一传动作和轨迹

    输入：
    1. cost_func: 目标函数
    2. initial_state: 初始状态
    4. trajectory: 轨迹（嵌套的List），使用multiprocess定义的共享变量
    5. 是否打印计算过程
    
    输出：
    1. 无输出，所有的轨迹被保存到共享变量trajectory中

    """
    trajectory_single = []
    state = initial_state
    for step in range(N):
        action, next_state, obj = qp_solver(cost_func, initial_state=state)
        trajectory_single.append((state, action))
        state = next_state
    trajectory.append(trajectory_single)
    if log:
        if len(trajectory) % 5 == 0:
            print("正在获取第{}个MPC最优解决，其损失为{}".format(len(trajectory), obj))


def mpcs_solver(cost_func, initial_states, process=os.cpu_count(), save=True):
    """
    功能：多进程求解多个MPC问题

    输入：
    1. cost_func: 目标函数
    2. N: 控制时域长度
    3. initial_states: 初始状态集合
    4. process: 默认为CPU核心数量，建议不要超过这个数字
    5. save: 是否保存：默认将轨迹保存到同目录下

    输出：
    1. trajectory: 轨迹集合
    """
    print("共有{}个MPC问题待求解".format(len(initial_states)))
    # 记录计算时间
    start_time = time.time()
    # 初始化轨迹池
    manager = multiprocessing.Manager()
    trajectory = manager.list()
    # for state in initial_states:
    #     mpc_solver(cost_func, state, trajectory)
    # 创建进程池
    pool = multiprocessing.Pool(process)
    for state in initial_states:
        pool.apply_async(mpc_solver, (cost_func, state, trajectory))
    pool.close()
    pool.join()
    if save:
        dump_path = open("trajectory.pickle", "wb")
        pickle.dump(trajectory, dump_path)
        dump_path.close()
    print("总共耗费时间：{}秒".format(time.time() - start_time))
    print(
        "平均每个MPC问题求解时间为：{}秒".format(
            float(time.time() - start_time) / len(initial_states)
        )
    )
    return trajectory

