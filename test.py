# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2022/11/26 09:43:08
@Author  :   olixu 
@Version :   1.0
@Contact :   273601727@qq.com
@WebSite    :   https://blog.oliverxu.cn
"""

# here put the import lib
import os
import numpy as np
import symengine
from cem_mpc_solver import mpcs_solver
from funcs import generate_cost_func
import pdb


cost_func = generate_cost_func()
# 定义步长
N = 10
# 定义初始状态集合
x_0 = np.arange(-2.0, 2.0, 0.25)
x_1 = np.arange(-2.0, 2.0, 0.25)
xx_0, xx_1 = np.meshgrid(x_0, x_1)  # 为一维的矩阵
initial_states = np.transpose(np.array([xx_0.ravel(), xx_1.ravel()]))

if __name__ == "__main__":
    # 调用求解器求解
    mpcs_solver(cost_func, initial_states, process=os.cpu_count(), save=True)

"""
cost_func = generate_cost_func()
initial_states = np.array([[-2.0, -2.0]])

if __name__ == "__main__":
    # 调用求解器求解
    mpcs_solver(cost_func, initial_states, process=os.cpu_count(), save=True)
"""

