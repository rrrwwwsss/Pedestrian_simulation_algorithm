# -*- coding: utf-8 -*-
"""
微信公众号：KAU的云实验台
"""
import random
import math
from solution import solution
import time
import numpy as np


def initialization(pop, ub, lb, dim):
    ''' 黏菌种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]  # 生成[lb,ub]之间的随机数

    return X

def SortFitness(Fit):
    '''适应度排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def SortPosition(X,index):
    '''根据适应度对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

def SMA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # 参数初始化
    bestPositions = np.zeros(dim)
    Destination_fitness = float("inf")
    fitness = np.zeros([SearchAgents_no, 1])
    W = np.zeros([SearchAgents_no, dim])
    z=0.03

    # 判断是否为向量
    if not isinstance(lb, list):
        # 向量化
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # 初始化种群
    X = initialization(SearchAgents_no, ub, lb, dim)  # 初始化种群
    # 更新个体与适应度
    for i in range(0, SearchAgents_no):
        # 边界检查
        X[i, :] = np.clip(X[i, :], lb, ub)
        # 适应度
        fitness[i] = objf(X[i, :])
        # 更新猎物位置
        if fitness[i] < Destination_fitness:
            Destination_fitness = fitness[i]
            bestPositions = X[i, :].copy()

    # 初始化收敛曲线
    convergence_curve = np.zeros(Max_iter-1)

    # 保存结果
    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 1  # Loop counter
    # 迭代
    while t < Max_iter:

        # 对适应度值排序
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)  # 种群排序
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S = bestFitness - worstFitness + 10E-8  # 当前最优适应度于最差适应度的差值，10E-8为极小值，避免分母为0；

        # 更新黏菌重量
        for i in range(SearchAgents_no):
            if i < SearchAgents_no / 2:
                W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
            else:
                W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)

        # 下式中的a、b分别是vb和vc中的参数
        tt = -(t / Max_iter) + 1
        if tt != -1 and tt != 1:
            a = math.atanh(tt)
        else:
            a = 1
        b = 1 - t / Max_iter

        # 核心更新机制
        for i in range(SearchAgents_no):
            if np.random.random() < z:
                # 分离部分个体搜索其他食物
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
            else:
                # 搜索食物
                p = np.tanh(abs(fitness[i] - Destination_fitness))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(SearchAgents_no)
                    B = np.random.randint(SearchAgents_no)
                    if r < p:
                        X[i, j] = bestPositions[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])  # 公式（1.4）第二个式子
                    else:
                        X[i, j] = vc[0, j] * X[i, j]

        # 更新个体与适应度
        for i in range(0, SearchAgents_no):
            # 边界检查
            X[i, :] = np.clip(X[i, :], lb, ub)
            # 适应度
            fitness[i] = objf(X[i, :])
            # 更新猎物位置
            if fitness[i] < Destination_fitness:
                Destination_fitness = fitness[i]
                bestPositions = X[i, :].copy()

        convergence_curve[t-1] = Destination_fitness
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(Destination_fitness)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "SMA"
    s.objfname = objf.__name__
    s.best = Destination_fitness
    s.bestIndividual = bestPositions

    return s
