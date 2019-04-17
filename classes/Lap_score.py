
from skfeature.function.similarity_based import lap_score
import numpy as np
from skfeature.utility import construct_W


class Lap_score_HSI(object):

    def __init__(self, n_band=10):
        self.n_band = n_band

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return:
        """
        # n_row, n_column, __n_band = X.shape
        # XX = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        XX = X

        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(XX, **kwargs_W)

        # obtain the scores of features
        score = lap_score.lap_score(X, W=W)

        # sort the feature scores in an ascending order according to the feature scores
        idx = lap_score.feature_ranking(score)

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:self.n_band]]

        # selected_features.reshape((self.n_band, n_row, n_column))
        # selected_features = np.transpose(selected_features, axes=(1, 2, 0))
        return selected_features
