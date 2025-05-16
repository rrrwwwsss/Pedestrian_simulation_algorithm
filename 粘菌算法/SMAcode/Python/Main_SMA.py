'''
# ----------------------- README -----------------------------------------
# -------------- 最后一次修改：2023/12/17 ----------------------------------
# -------------------  欢迎关注₍^.^₎♡  ------------------------------------
# -------------- 项目：黏菌算法(SMA) --------------------------------
# -------------- 微信公众号：KAU的云实验台 ----------------------------------
# -------------- CSDN：KAU的云实验台 ---------------------------------------
# -------------- 付费代码(更全)：https://mbd.pub/o/author-a2iWlGtpZA== -----
# -------------- 免费代码：公众号后台回复"资源" -----------------------------
# ------------------------------------------------------------------------
'''

import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import benchmarks
import select_Optimize_algorithm

''' --------------------------- 参数设置 ----------------------------------'''
SearchAgents_no = 30  # 种群量级
Function_name = 'F13'  # 测试函数
Max_iteration = 500   # 迭代次数

''' ------------------------ 获取测试函数细节 F1~F23 ----------------------------------'''
func_details = benchmarks.getFunctionDetails(Function_name)
lb = func_details[1]
ub = func_details[2]
dim = func_details[3]
fobj = getattr(benchmarks, Function_name) # 获取函数求解
print(func_details)

''' ------------------------ 黏菌求解 ----------------------------------'''
x = select_Optimize_algorithm.SMA(fobj, lb, ub, dim, SearchAgents_no, Max_iteration)

''' ------------------------ 求解结果 ----------------------------------'''
IterCurve = x.convergence
Best_fitness = x.best
Best_Pos = x.bestIndividual

''' ------------------------ 绘图 ----------------------------------'''
part1 = ['SMA',Function_name]
name1 = ' '.join(part1)
plt.figure(1)
plt.plot(IterCurve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title(name1, fontsize='large')
label = [name1]
plt.legend(label, loc='upper right')
plt.savefig('./SMA_Python.jpg')
# plt.show()







