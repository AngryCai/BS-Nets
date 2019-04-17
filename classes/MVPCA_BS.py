# -*- coding: utf-8 -*-
"""
@ Description: 
-------------
HSI band selection using PCA transform
-------------
@ Time    : 2018/11/20 11:17
@ Author  : Yaoming Cai
@ FileName: MVPCA_BS.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np


class MVPCA:

    def __init__(self, n_band):
        self.n_band = n_band

    def fit(self, X, y=None):
        """
        :param X: 2-D image with shape (n_pixel, n_band)
        :param y:
        :return:
        """
        self.X = X
        return self

    def predict(self, X):
        x_mean = np.mean(X, axis=0)
        x_new = X - x_mean
        conv_mat = np.cov(x_new, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(conv_mat)
        eig_val = eig_val ** .5
        mat_var = eig_vec * eig_val
        sum_var = np.sum(mat_var ** 2, axis=1)
        band_indx = np.argsort(sum_var)[-self.n_band:]
        x_new = X[:, band_indx]
        print('MVPCA band index:', band_indx)
        return x_new
