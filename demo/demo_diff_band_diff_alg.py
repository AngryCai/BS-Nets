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

import sys
sys.path.append('/home/caiyaom/python_codes/')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import maxabs_scale, minmax_scale
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

from DeepLearning.AttentionBandSelection.classes.Attention_CNN_BS import Attention_BS
from Toolbox.Preprocessing import Processor
from DeepLearning.AttentionBandSelection.classes.utility import eval_band_cv
from DeepLearning.AttentionBandSelection.classes.ISSC import ISSC_HSI
from DeepLearning.AttentionBandSelection.classes.SNMF import BandSelection_SNMF
from DeepLearning.AttentionBandSelection.classes.SpaBS import SpaBS
from DeepLearning.AttentionBandSelection.classes.MVPCA_BS import MVPCA
from DeepLearning.AttentionBandSelection.classes.Lap_score import Lap_score_HSI
from DeepLearning.AttentionBandSelection.classes.SSR import SSC_BS


# path = 'C:\\Users\\07\Desktop\putty\history.npz'
# path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-Salinas-KNN-SVM-multiscale-5band.npz'
path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-PaviaU-Conv-att-100epoch-5band-best.npz'
npz = np.load(path)
# s = npz['score']
# s1 = npz['score'][:, 0, 0, 0]
# s2 = npz['score'][:, 1, 0, 0]
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
acc_dict = {'Attention_BS':[], 'ISSC':[], 'SpaBS':[], 'MVPCA':[], 'SNMF':[]}
band_range = np.arange(3, 30, 2)
# alg_key = ['Attention_BS', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
alg_key = ['Attention_BS']
for i in band_range:
    alg = {
        'Attention_BS': Attention_BS(w, i),
        # 'ISSC': ISSC_HSI(i, coef_=1e5),
        # 'SpaBS': SpaBS(i),
        # 'MVPCA': MVPCA(i),
        # 'SNMF': BandSelection_SNMF(i)
    }
    for j in range(len(alg_key)):
        if alg_key[j] == 'SpaBS' and i == band_range[0]:
            x_new = alg[alg_key[j]].predict(X_img_2D)
            gamma = alg[alg_key[j]].gamma
        elif alg_key[j] == 'SpaBS' and i > band_range[0]:
            x_new = alg[alg_key[j]].predict_gamma(X_img_2D, gamma)
        else:
            x_new = alg[alg_key[j]].predict(X_img_2D)
        x_new_3D = x_new.reshape((n_row, n_column, x_new.shape[-1]))
        img_correct, gt_correct = p.get_correct(x_new_3D, gt)
        score = eval_band_cv(img_correct, gt_correct, times=20)
        # acc_list.append(score)
        acc_dict[alg_key[j]].append(score)
        print('%s band, alg: %s: %s' % (i, alg_key[j], score))

# # all band
# img_correct, gt_correct = p.get_correct(X_img, gt)
# score_all = eval_band_cv(img_correct, gt_correct, times=20)
# print('ALL BAND:', score_all)
np.savez('../results/diff_band_diff_alg_acc.npz', res=acc_dict)
# np.savez('../results/diff_band_diff_alg_acc.npz', res=acc_dict, all_band=score_all)


"""
=====================
 PLOT acc v.s diff-bands
=====================
"""

"""
import matplotlib.pyplot as plt
import numpy as np
FONTSIZE = 12
LINEWIDTH = 1.8
MARKERSIZE = 8
CLASSIFIER = 'svm'  # 'knn'
INDICATOR = 'oa'  # 'kappa'
p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_diff_alg-acc-IndianPines-[3-30]-best.npz'
a = np.load(p)
score = a['res'][()]
alg_key = ['Attention_BS', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
plot_key = ['BS-Net', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
marker = ['o', 'v', 's', 'd', 'H', '*']
fig, ax = plt.subplots()
x_ = np.arange(3, 30, 2)
j = 0
lines = []
for k in alg_key:
    s = []
    for i in range(len(x_)):
        s.append(score[k][i][CLASSIFIER][INDICATOR][0])
    line, = ax.plot(x_, s, linestyle='-', marker=marker[j], markerfacecolor='None', linewidth=LINEWIDTH, markersize=MARKERSIZE, label=plot_key[j])
    lines.append(line)
    j += 1

# # plot all band
s_all = a['all_band'][()][CLASSIFIER][INDICATOR][0]
line, = ax.plot(x_, [s_all]*len(x_), linestyle='--', markerfacecolor='None', linewidth=LINEWIDTH, markersize=MARKERSIZE, label='All bands')
lines.append(line)

plt.legend(handles=lines, loc=4, prop={'size': FONTSIZE}, shadow=True)
ymin, ymax = plt.ylim()
plt.ylim(ymin-ymin/2, ymax)
ax.set_xlabel('Number of selected bands', fontsize=FONTSIZE)
# Make the y-axis label, ticks and tick labels match the line color.
ax.set_ylabel('OA (%)', fontsize=FONTSIZE)
ax.tick_params('y', labelsize=FONTSIZE)
ax.tick_params('x', labelsize=FONTSIZE)
plt.grid(True)
plt.show()
"""

"""
=========================
print acc
=========================
"""
# import numpy as np
# FONTSIZE = 12
# LINEWIDTH = 1.8
# MARKERSIZE = 8
# CLASSIFIER = 'svm'  # 'knn'
# INDICATOR = 'oa'  # 'kappa'
# p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_acc-IndianPines-3-30.npz'
# a = np.load(p)
# score = a['res'][()]
# alg_key = ['Attention_BS', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
# # alg_key = ['Attention_BS', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS']
# for k in alg_key:
#     print('======================================')
#     print(k)
#     print(score[k][9][CLASSIFIER])


"""
=========================
print train-test number
=========================
"""
# import numpy as np
# from sklearn.model_selection import cross_val_score, train_test_split
# from Toolbox.Preprocessing import Processor
# root = 'F:\Python\HSI_Files\\'
# # root = '/home/caiyaom/HSI_Files/'
# # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# # im_, gt_ = 'Pavia', 'Pavia_gt'
# # im_, gt_ = 'PaviaU', 'PaviaU_gt'
# # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# # im_, gt_ = 'Botswana', 'Botswana_gt'
# # im_, gt_ = 'KSC', 'KSC_gt'
#
# img_path = root + im_ + '.mat'
# gt_path = root + gt_ + '.mat'
# print(img_path)
# p = Processor()
# img, gt = p.prepare_data(img_path, gt_path)
#
# img_correct, gt_correct = p.get_correct(img, gt)
# X_train, X_test, y_train, y_test = train_test_split(img_correct, gt_correct, test_size=0.95, random_state=None, shuffle=True, stratify=gt_correct)
# for i in np.unique(y_train):
#     n_tr = len(np.nonzero(y_train == i)[0])
#     n_ts = len(np.nonzero(y_test == i)[0])
#     print('%s-th class: %s, %s' % (i, n_tr, n_ts))


"""
======================
Print Band index
======================
"""
# import numpy as np
# path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-Indian-Conv-att-100epoch-5band-best.npz'
# npz = np.load(path)
# # s = npz['score']
# # s1 = npz['score'][:, 0, 0, 0]
# # s2 = npz['score'][:, 1, 0, 0]
# w = npz['channel_weight'][-1]
# mean_weight = np.mean(w, axis=0)
# band_indx = np.argsort(mean_weight)[-50:]
# i = 0
# for j in band_indx[-1::-1]:
#     print(j, '/')
#     if (i+1) % 10 == 0:
#         print('==============')
#     i += 1


"""
========================================
Plot band weight contribution 
========================================
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
FONTSIZE = 12
LINEWIDTH = 1.8

path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-PaviaU-Conv-att-100epoch-5band-best.npz'
npz = np.load(path)
# s = npz['score']
# s1 = npz['score'][:, 0, 0, 0]
# s2 = npz['score'][:, 1, 0, 0]
w = npz['channel_weight'][-1]
mean_weight = np.mean(w, axis=0)
mean_weight = mean_weight/mean_weight.sum()
band_indx = np.argsort(mean_weight)
sort_band_weight = mean_weight[band_indx][-1::-1]
s = 0.
raio = []
for w_i in sort_band_weight:
    s += w_i
    raio.append(s)
fig, axes = plt.subplots(1, 1)
axes.plot(raio, linewidth=LINEWIDTH)
axes.set_xlabel('Number of bands', fontsize=FONTSIZE)
axes.set_ylabel('Ratio', fontsize=FONTSIZE)
axes.tick_params('y', labelsize=FONTSIZE)
axes.tick_params('x', labelsize=FONTSIZE)
# plt.xticks(range(5), band_indx[:5], rotation=60)
# plt.yticks(raio[:10], np.round(raio[:10], decimals=2))
plt.grid()

inset_axes = inset_axes(axes,
                        width="50%",  # width = 50% of parent_bbox
                        height="40%",  # "40% of parent_bbox
                        loc=7)
# inset_axes
inset_axes.plot(raio[:10], linewidth=LINEWIDTH)
plt.xticks(range(10), band_indx[-1::-1][:10], rotation=60)
plt.yticks(raio[:10], np.round(raio[:10], decimals=2))
plt.xlabel('Band index')
plt.ylabel('Ratio')
# inset_axes.tick_params('y', labelsize=6)
# inset_axes.tick_params('x', labelsize=6)
plt.grid()
plt.show()


"""
=====================
show spectral band with best bands
=====================
"""
import matplotlib.pyplot as plt
import numpy as np
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import maxabs_scale, minmax_scale
root = 'F:\Python\HSI_Files\\'
# root = '/home/caiyaom/HSI_Files/'
# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'Pavia', 'Pavia_gt'
# im_, gt_ = 'PaviaU', 'PaviaU_gt'
# im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'KSC', 'KSC_gt'
img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'
FONTSIZE = 12
print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
n_row, n_column, n_band = img.shape
img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
# indina: [46, 33, 140, 161, 80, 35, 178, 44, 126, 36] # class [1, 2, 3, 9, 10, 11]
#  PaviaU: [90, 42, 16, 48, 71, 3, 78, 38, 80, 53]
#  Salinas: [93, 123, 68, 32, 76, 106, 90, 7, 173, 67]
path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-Indian-Conv-att-100epoch-5band-best.npz'
npz = np.load(path)
# s = npz['score']
# s1 = npz['score'][:, 0, 0, 0]
# s2 = npz['score'][:, 1, 0, 0]
w = npz['channel_weight'][-1]
mean_weight = np.mean(w, axis=0)
mean_weight = mean_weight/mean_weight.sum()
band_indx = np.argsort(mean_weight)[-1::-1][:5]
sort_band_weight = mean_weight[band_indx]

# band_indx = [46, 33, 140, 161, 80, 35, 178, 44, 126, 36]
label = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees']
# label = ['Alphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets']
# label = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth']

linestyle = ['-', '--', ':', '-:']
fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(5):  # [1, 3, 6, 9]:
    c_inx = np.nonzero(i == gt)
    x = img[c_inx[0][15], c_inx[1][15], :]  # 10th pixel
    plt.plot(x, linewidth=1.8, label=label[i])
    # plt.scatter(band_indx, x[band_indx], marker='o')
plt.xticks(np.arange(0, n_band)[band_indx], band_indx)
axes.set_xlabel('Spectral band', fontsize=FONTSIZE)
axes.set_ylabel('Reflectance', fontsize=FONTSIZE)
axes.tick_params('y', labelsize=FONTSIZE)
axes.tick_params('x', labelsize=FONTSIZE)
plt.legend(loc='best')
for j, w in zip(band_indx, sort_band_weight):
    plt.axvline(j, linestyle='--', linewidth=w * 20, color='red', alpha=0.5)
plt.show()




