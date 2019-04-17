# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2019/3/6 15:45
@ Author  : Yaoming Cai
@ FileName: demo-MOBS-bandindx-from-matlab.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
from scipy.io import loadmat

path = ['F:\Python\DeepLearning\AttentionBandSelection\\results\Indian_pareto.mat',
 'F:\Python\DeepLearning\AttentionBandSelection\\results\PaviaU_pareto.mat',
'F:\Python\DeepLearning\AttentionBandSelection\\results\Salinas_pareto.mat']
list_ = []
for p in path:
    mat = loadmat(p)
    pareto = mat['pareto']
    pareto = pareto.tolist()
    indx_list = []
    for i in range(50):
        code_ = pareto[i][0][0]
        code = code_.reshape(-1)
        index = np.nonzero(code)[0]
        indx_list.append(index)
        print(index)
    print('=========================================')
    list_.append(indx_list)
np.savez('F:\Python\DeepLearning\AttentionBandSelection\\results\MOBS-indian-paviaU-salinas-top50band-index',
         Indian=list_[0], PaviaU=list_[1], Salinas=list_[2])
