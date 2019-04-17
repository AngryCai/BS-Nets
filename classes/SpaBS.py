# coding:utf-8
# @ Author by Zeng Meng

import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram


class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        if min(X.shape) <= self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


class SpaBS(object):

    def __init__(self, n_band, sparsity_level=0.5):
        self.n_band = n_band
        self.sparsity_level = sparsity_level

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """
        Select band according to sparse representation
        :param X: array like: shape (n_row*n_column, n_band)
        :return:
        """
        # n_row, n_column, n_band = X.shape
        # XX = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        # 使用SpaBS算法
        # 调用ksvd
        # TODO: according to ref., X has to be with shape (n_band, n_sample)
        # X = X.transpose()
        dico = ApproximateKSVD(n_components=X.shape[1])
        dico.fit(X)
        gamma_ = dico.transform(X)  # gamma为系数矩阵, shape(n_sample, n_atom)
        gamma = gamma_.transpose()
        self.gamma = gamma
        sorted_inx = np.argsort(gamma, axis=0)  # ascending order for each column
        K = X.shape[0] * self.sparsity_level
        largest_k = sorted_inx[-self.n_band:, :]

        # # statistic
        element, freq = np.unique(largest_k, return_counts=True)
        selected_inx = element[np.argsort(freq)][-self.n_band:]
        print('SpaBS band index:', selected_inx)
        selected_band = X[:, selected_inx]
        return selected_band

    def predict_gamma(self, X, gamma):
        sorted_inx = np.argsort(gamma, axis=0)  # ascending order for each column
        largest_k = sorted_inx[-self.n_band:, :]

        # # statistic
        element, freq = np.unique(largest_k, return_counts=True)
        selected_inx = element[np.argsort(freq)][-self.n_band:]
        print('SpaBS band index:', selected_inx)
        selected_band = X[:, selected_inx]
        return selected_band



'''
---------------------------
        Test
'''

# X ~ gamma.dot(dictionary)
# X = np.random.randn(1000, 20)
# aksvd = ApproximateKSVD(n_components=20)
# dictionary = aksvd.fit(X).components_
# gamma = aksvd.transform(X)
# print(gamma.shape)