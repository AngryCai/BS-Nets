"""
Description:
    auxiliary functions
"""
from Toolbox.Preprocessing import Processor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


def eval_band(new_img, gt, train_inx, test_idx):
    """

    :param new_img:
    :param gt:
    :param train_inx:
    :param test_idx:
    :return:
    """
    p = Processor()
    # img_, gt_ = p.get_correct(new_img, gt)
    gt_ = gt
    img_ = maxabs_scale(new_img)
    # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
    X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
    knn_classifier = KNN(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
    y_pre = knn_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pre)
    # score = np.mean(score)
    return score


def eval_band_cv(X, y, times=10, test_size=0.95):
    p = Processor()
    estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
    estimator_pre, y_test_all = [[], []], []
    for i in range(times):  # repeat N times K-fold CV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=None, shuffle=True, stratify=y)
        # skf = StratifiedKFold(n_splits=20, shuffle=True)
        # for test_index, train_index in skf.split(img_correct, gt_correct):
        #     X_train, X_test = img_correct[train_index], img_correct[test_index]
        #     y_train, y_test = gt_correct[train_index], gt_correct[test_index]
        y_test_all.append(y_test)
        for c in range(len(estimator)):
            estimator[c].fit(X_train, y_train)
            y_pre = estimator[c].predict(X_test)
            estimator_pre[c].append(y_pre)
    # score = []
    score_dic = {'knn':{'ca':[], 'oa':[], 'aa':[], 'kappa':[]},
                 'svm': {'ca': [], 'oa': [], 'aa': [], 'kappa': []}
                 }
    key_ = ['knn', 'svm']
    for z in range(len(estimator)):
        ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=None, verbose=False)
        # score.append([oa, kappa, aa, ca])
        score_dic[key_[z]]['ca'] = ca
        score_dic[key_[z]]['oa'] = oa
        score_dic[key_[z]]['aa'] = aa
        score_dic[key_[z]]['kappa'] = kappa
    return score_dic

def cal_mean_spectral_divergence(band_subset):
    """
    Spectral Divergence is defined as the symmetrical KL divergence (D_KLS) of two bands probability distribution.
    We use Mean SD (MSD) to quantify the redundancy among a band set.

    B_i and B_j should be a gray histagram.
    SD = D_KL(B_i||B_j) + D_KL(B_j||B_i)
    MSD = 2/n*(n-1) * sum(ID_ij)

    Ref:
    [1]	GONG MAOGUO, ZHANG MINGYANG, YUAN YUAN. Unsupervised Band Selection Based on Evolutionary Multiobjective
    Optimization for Hyperspectral Images [J]. IEEE Transactions on Geoscience and Remote Sensing, 2016, 54(1): 544-57.

    :param band_subset: with shape (n_row, n_clm, n_band)
    :return:
    """
    n_row, n_column, n_band = band_subset.shape
    N = n_row * n_column
    hist = []
    for i in range(n_band):
        hist_, edge_ = np.histogram(band_subset[:, :, i], 256)
        hist.append(hist_ / N)
    hist = np.asarray(hist)
    hist[np.nonzero(hist <= 0)] = 1e-20
    # entropy_lst = entropy(hist.transpose())
    info_div = 0
    # band_subset[np.nonzero(band_subset <= 0)] = 1e-20
    for b_i in range(n_band):
        for b_j in range(n_band):
            band_i = hist[b_i].reshape(-1)/np.sum(hist[b_i])
            band_j = hist[b_j].reshape(-1)/np.sum(hist[b_j])
            entr_ij = entropy(band_i, band_j)
            entr_ji = entropy(band_j, band_i)
            entr_sum = entr_ij + entr_ji
            info_div += entr_sum
    msd = info_div * 2 / (n_band * (n_band - 1))
    return msd

def cal_mean_spectral_angle(band_subset):
    """
    Spectral Angle (SA) is defined as the angle between two bands.
    We use Mean SA (MSA) to quantify the redundancy among a band set.
    i-th band B_i, and j-th band B_j,
    SA = arccos [B_i^T * B_j / ||B_i|| * ||B_j||]
    MSA = 2/n*(n-1) * sum(SA_ij)

    Ref:
    [1]	GONG MAOGUO, ZHANG MINGYANG, YUAN YUAN. Unsupervised Band Selection Based on Evolutionary Multiobjective
    Optimization for Hyperspectral Images [J]. IEEE Transactions on Geoscience and Remote Sensing, 2016, 54(1): 544-57.

    :param band_subset: with shape (n_row, n_clm, n_band)
    :return:
    """
    n_row, n_column, n_band = band_subset.shape
    spectral_angle = 0
    for i in range(n_band):
        for j in range(n_band):
            band_i = band_subset[i].reshape(-1)
            band_j = band_subset[j].reshape(-1)
            lower = np.sum(band_i ** 2) ** 0.5 * np.sum(band_j ** 2) ** 0.5
            higher = np.dot(band_i, band_j)
            if higher / lower > 1.:
                angle_ij = np.arccos(1. - 1e-16)
                # print('1-higher-lower', higher - lower)
            # elif higher / lower < -1.:
            #     angle_ij = np.arccos(1e-8 - 1.)
                # print('2-higher-lower', higher - lower)
            else:
                angle_ij = np.arccos(higher / lower)
            spectral_angle += angle_ij
    msa = spectral_angle * 2 / (n_band * (n_band - 1))
    return msa