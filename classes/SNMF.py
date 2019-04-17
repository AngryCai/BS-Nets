# coding:utf-8

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nimfa
from sklearn.metrics import accuracy_score

class BandSelection_SNMF(object):
    def __init__(self, n_band):
        self.n_band = n_band

    def predict(self, X):
        """
        :param X: with shape (n_pixel, n_band)
        :return:
        """
        # # Note that X has to reshape to (n_fea., n_sample)
        # XX = X.transpose()  # (n_band, n_pixel)
        # snmf = nimfa.Snmf(X, seed="random_c", rank=self.n_band)  # remain para. default
        snmf = nimfa.Snmf(X, rank=self.n_band, max_iter=30, version='r', eta=1.,
                          beta=1e-4, i_conv=10, w_min_change=0, rcond=None)
        snmf_fit = snmf()
        W = snmf.basis()  # shape: n_band * k
        H = snmf.coef()  # shape: k * n_pixel

        #  get clustering res.
        H = np.asarray(H)
        indx_sort = np.argsort(H, axis=0)  # ascend order
        cluster_res = indx_sort[-1].reshape(-1)

        #  select band
        selected_band = []
        for c in np.unique(cluster_res):
            idx = np.nonzero(cluster_res == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        while selected_band.__len__() < self.n_band:
            selected_band.append(np.zeros(X.shape[0]))
        bands = np.asarray(selected_band).transpose()
        return bands