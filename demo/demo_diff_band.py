# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/11/26 19:44
@ Author  : Yaoming Cai
@ FileName: demo_diff_band.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import maxabs_scale, minmax_scale
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

from DeepLearning.AttentionBandSelection.classes.Attention_CNN_BS import Attention_BS
from Toolbox.Preprocessing import Processor


def eval_band_cv(X, y, times=10, test_size=0.95):
    estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
    estimator_pre, y_test_all = [[], []], []
    for i in range(times):  # repeat N times K-fold CV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None, shuffle=True)
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


# path = 'C:\\Users\\07\Desktop\putty\history.npz'
path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-PaviaU-Conv-att-100epoch-5band-best.npz'
npz = np.load(path)
# s = npz['score']
s1 = npz['score'][:, 0, 0, 0]
s2 = npz['score'][:, 1, 0, 0]
loss_ = npz['loss']
w = npz['channel_weight'][-1]

root = 'F:\Python\HSI_Files\\'
# root = '/home/caiyaom/HSI_Files/'
# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'Pavia', 'Pavia_gt'
im_, gt_ = 'PaviaU', 'PaviaU_gt'
# im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'
print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
# Img, Label = Img[:256, :, :], Label[:256, :]
n_row, n_column, n_band = img.shape
X_img = minmax_scale(img.reshape((n_row * n_column, n_band))).reshape((n_row, n_column, n_band))
X_img_2D = X_img.reshape((n_row * n_column, n_band))
gt_1D = gt.reshape(-1)
acc_list = []
band_range = np.arange(1, 30, 2)
for i in band_range:
    abs_ = Attention_BS(w, i)
    x_new = abs_.predict(X_img_2D)
    x_new_3D = x_new.reshape((n_row, n_column, x_new.shape[-1]))
    img_correct, gt_correct = p.get_correct(x_new_3D, gt)
    score = eval_band_cv(img_correct, gt_correct, times=20)
    acc_list.append(score)
    print('%s band: %s' % (i, score))
np.savez('../results/diff_band_acc.npz', res=acc_list)

# # all band
img_correct, gt_correct = p.get_correct(X_img, gt)
score = eval_band_cv(img_correct, gt_correct, times=20)
print('ALL BAND:', score)

"""
=====================
        PLOT 
=====================
"""
import matplotlib.pyplot as plt
import numpy as np
p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_acc.npz'
a = np.load(p)
score = a['res']
s = []
for i in range(len(score)):
    s.append(score[i]['svm']['oa'][0])
plt.plot(s)