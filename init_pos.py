# encoding: utf-8
import random
import numpy as np
import archiving_pos
import pareto_pos


def init_designparams(particals, in_min, in_max):
    in_dim = len(in_max)  # 粒子维度
    in_temp = np.zeros((particals, in_dim))  # 行数：粒子个数  列数：粒子纬度
    for i in range(particals):  # 遍历每个粒子的每个属性
        for j in range(in_dim):
            in_temp[i, j] = random.uniform(0, 1) * (in_max[j] - in_min[j]) + in_min[j]
    return in_temp


def init_v(particals, v_max, v_min):
    v_dim = len(v_max)  # 输入参数的维度
    v_ = np.zeros((particals, v_dim))
    for i in range(particals):
        for j in range(v_dim):
            v_[i, j] = random.uniform(0, 1) * (v_max[j] - v_min[j]) + v_min[j]
    return v_


def init_pbest(in_, fitness_, syn=None):
    return in_, fitness_, syn


def init_archive(in_, fitness_, syn_=None):  # in_: 所有粒子位置 fitness_: 所有粒子适应值
    pareto_c = pareto_pos.Pareto_(in_, fitness_, syn_)
    curr_archiving_in, curr_archiving_fit, curr_archiving_syn = pareto_c.pareto()
    return curr_archiving_in, curr_archiving_fit, curr_archiving_syn


def init_gbest(curr_archiving_in, curr_archiving_fit, curr_archiving_syn, mesh_div, min_, max_, particals):
    get_g = archiving_pos.get_gbest(curr_archiving_in, curr_archiving_fit,curr_archiving_syn,
                                    mesh_div, min_, max_, particals)
    return get_g.get_gbest()
