# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/12/12 10:22
@ Author  : Yaoming Cai
@ FileName: FDPC_BS.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
from pydpc import Cluster
from pydpc._reference import Cluster as RefCluster

class FDPC_BC:

    def __init__(self):
        pass

    def fit(self, X, y=None):
        clu = Cluster(X, fraction=0.02, autoplot=False)
        clu.assign(20, 1.5)
        res = clu.membership

    def predict(self, X):
        return 
