# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/11/12 14:26
@ Author  : Yaoming Cai
@ FileName: Attention_CNN_loss_acc_temp.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import sys
sys.path.append('/home/caiyaom/python_codes/')
import numpy as np


class Attention_BS:

    def __init__(self, channel_weight, n_band=10):
        self.n_band = n_band
        self.channel_weight = channel_weight

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        mean_weight = np.mean(self.channel_weight, axis=0)
        band_indx = np.argsort(mean_weight)[::-1][:self.n_band]  # # from larger to smaller
        print('=============================')
        print('SELECTED BAND: ', band_indx)
        print('=============================')
        X_new = X[:, band_indx]
        return X_new
