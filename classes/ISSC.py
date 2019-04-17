"""
Ref:
    [1]	W. Sun, L. Zhang, B. Du, W. Li, and Y. Mark Lai, "Band Selection Using Improved Sparse Subspace Clustering
    for Hyperspectral Imagery Classification," IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing, vol. 8, pp. 2784-2797, 2015.

Formula:
    arg min ||X - XW||_F + lambda||W||_F subject to diag(Z) = 0
Solution:
    Wˆ = −(X^T X + lambda*I)^−1 (diag((X^T X + lambda*I)−1))^−1
"""

import numpy as np
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from sklearn.cluster.spectral import SpectralClustering


class ISSC_HSI(object):
    """
    :argument:
        Implementation of L2 norm based sparse self-expressive clustering model
        with affinity measurement basing on angular similarity
    """
    def __init__(self, n_band=10, coef_=1):
        self.n_band = n_band
        self.coef_ = coef_

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return: selected band subset
        """
        I = np.eye(X.shape[1])
        coefficient_mat = -1 * np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.coef_ * I),
                                      np.linalg.inv(np.diag(np.diag(np.dot(X.transpose(), X) + self.coef_ * I))))
        temp = np.linalg.norm(coefficient_mat, axis=0).reshape(1, -1)
        affinity = (np.dot(coefficient_mat.transpose(), coefficient_mat) /
                    np.dot(temp.transpose(), temp))**2

        sc = SpectralClustering(n_clusters=self.n_band, affinity='precomputed')
        sc.fit(affinity)
        selected_band = self.__get_band(sc.labels_, X)
        return selected_band

    def __get_band(self, cluster_result, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        selected_band = []
        n_cluster = np.unique(cluster_result).__len__()
        # img_ = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band).transpose()
        # bands = bands.reshape(n_cluster, n_row, n_column)
        # bands = np.transpose(bands, axes=(1, 2, 0))
        return bands