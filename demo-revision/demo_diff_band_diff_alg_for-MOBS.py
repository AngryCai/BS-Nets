# -*- coding: utf-8 -*-
"""
@ Description:
-------------
Test classification acc for only FC-Attention BS
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
path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\MOBS-indian-paviaU-salinas-top50band-index.npz'
npz = np.load(path)
band_indx_list = npz['Salinas']

root = 'F:\Python\HSI_Files\\'
# root = '/home/caiyaom/HSI_Files/'
# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
# im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'Pavia', 'Pavia_gt'
# im_, gt_ = 'PaviaU', 'PaviaU_gt'
im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
# im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'
print(img_path)
processor = Processor()
img, gt = processor.prepare_data(img_path, gt_path)
# Img, Label = Img[:256, :, :], Label[:256, :]
n_row, n_column, n_band = img.shape
X_img = minmax_scale(img.reshape((n_row * n_column, n_band))).reshape((n_row, n_column, n_band))
X_img_2D = X_img.reshape((n_row * n_column, n_band))
gt_1D = gt.reshape(-1)
acc_list = []
acc_dict = {'MOBS':[]}
band_range = np.arange(3, 30, 2)
# alg_key = ['Attention_BS', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
alg_key = ['MOBS']
for i in band_range:
    for j in range(len(alg_key)):
        x_new = X_img[:, :, band_indx_list[i - 1]]
        x_new_3D = x_new.reshape((n_row, n_column, x_new.shape[-1]))
        img_correct, gt_correct = processor.get_correct(x_new_3D, gt)
        score = eval_band_cv(img_correct, gt_correct, times=20)
        # acc_list.append(score)
        acc_dict[alg_key[j]].append(score)
        print('%s band, alg: %s: %s' % (i, alg_key[j], score))

# # all band
# img_correct, gt_correct = p.get_correct(X_img, gt)
# score_all = eval_band_cv(img_correct, gt_correct, times=20)
# print('ALL BAND:', score_all)
# np.savez('../results/diff_band_diff_alg_acc_MOBS-Indian.npz', res=acc_dict)
# np.savez('../results/diff_band_diff_alg_acc.npz', res=acc_dict, all_band=score_all)


"""
======================
add MOBS res. into FC.
======================
"""

import matplotlib.pyplot as plt
import numpy as np
p_conv = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_diff_alg-acc-Salinas-[3-30]-added-FC.npz'
a_conv = np.load(p_conv)
score_conv = a_conv['res'][()]

# p_mlp = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_diff_alg_acc_FC_Attention-Salinas.npz'
# a_mlp = np.load(p_mlp)
# score_mlp = a_mlp['res'][()]['Attention_BS_FC']

score_conv['MOBS'] = acc_dict['MOBS']
np.savez('F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_diff_alg-acc-Salinas-[3-30]-added-FC-MOBS.npz',
         res=score_conv, all_band=a_conv['all_band'][()])


"""
=====================
 PLOT acc v.s diff-bands with FC
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
p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_band_diff_alg-acc-PaviaU-[3-30]-added-FC-MOBS.npz'
a = np.load(p)
score = a['res'][()]
alg_key = ['Attention_BS', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS']
plot_key = ['BS-Net-Conv', 'BS-Net-FC', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS']
marker = ['o', 'v', 's', 'd', 'H', '*', '^', 'D']
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