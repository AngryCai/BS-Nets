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
from sklearn.preprocessing import minmax_scale

from DeepLearning.AttentionBandSelection.classes.Attention_CNN_BS import Attention_BS
from Toolbox.Preprocessing import Processor
from DeepLearning.AttentionBandSelection.classes.ISSC import ISSC_HSI
from DeepLearning.AttentionBandSelection.classes.SNMF import BandSelection_SNMF
from DeepLearning.AttentionBandSelection.classes.SpaBS import SpaBS
from DeepLearning.AttentionBandSelection.classes.MVPCA_BS import MVPCA
import time


def get_index(selected_band, raw_HSI):
    """
    :param selected_band: 3-D cube
    :param raw_HSI: 3-D cube
    :return:
    """
    band_index = []
    for i in range(selected_band.shape[-1]):
        band_i = np.reshape(selected_band[:, :, i], (selected_band.shape[0], selected_band.shape[1], 1))
        band_index.append(np.argmin(np.sum(np.abs(raw_HSI - band_i), axis=(0, 1))))
    return np.asarray(band_index)



path_conv = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-Indian-Conv-att-500epoch-5band-best.npz'
path_mlp = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-MLP-Indian-100epoch-5band.npz'
npz_mlp = np.load(path_mlp)
loss_mlp = npz_mlp['loss']
w_mlp = npz_mlp['channel_weight'][-1]

npz_conv = np.load(path_conv)
loss_conv = npz_conv['loss']
w_conv = npz_conv['channel_weight'][-1]

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
print(img_path)
p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
# Img, Label = Img[:256, :, :], Label[:256, :]
n_row, n_column, n_band = img.shape
X_img = minmax_scale(img.reshape((n_row * n_column, n_band))).reshape((n_row, n_column, n_band))
X_img_2D = X_img.reshape((n_row * n_column, n_band))
gt_1D = gt.reshape(-1)
acc_list = []
acc_dict = {'Attention_BS_Conv': {'top_50_bands': None, 'time': None},
            'Attention_BS_FC': {'top_50_bands': None, 'time': None},
            'ISSC': {'top_50_bands': None, 'time': None},
            'SpaBS': {'top_50_bands': None, 'time': None},
            'MVPCA': {'top_50_bands': None, 'time': None},
            'SNMF': {'top_50_bands': None, 'time': None}
            }

alg_key = ['Attention_BS_Conv', 'Attention_BS_FC', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
n_bands_selected = 50
alg = {
        'Attention_BS_Conv': Attention_BS(w_conv, n_bands_selected),
        'Attention_BS_FC': Attention_BS(w_mlp, n_bands_selected),
        'ISSC': ISSC_HSI(n_bands_selected, coef_=1e5),
        'SpaBS': SpaBS(n_bands_selected),
        'MVPCA': MVPCA(n_bands_selected),
        'SNMF': BandSelection_SNMF(n_bands_selected)
    }

for j in range(len(alg_key)):
    time_start = time.clock()
    x_new = alg[alg_key[j]].predict(X_img_2D)
    run_time = round(time.clock() - time_start, 3)

    # get band index
    x_new_3D = x_new.reshape((n_row, n_column, x_new.shape[-1]))
    band_index = get_index(x_new_3D, X_img)
    # acc_list.append(score)
    acc_dict[alg_key[j]]['top_50_bands'] = band_index
    acc_dict[alg_key[j]]['time'] = run_time
    print('alg: %s\n time= %s\n band_index:%s ' % (alg_key[j], run_time, band_index.tolist()))

# # all band
# img_correct, gt_correct = p.get_correct(X_img, gt)
# score_all = eval_band_cv(img_correct, gt_correct, times=20)
# print('ALL BAND:', score_all)
np.savez('../results/diff_alg_bandindex_runtime-PaviaU.npz', res=acc_dict)

# p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_alg_bandindex_runtime-IndianPines.npz'
# npz = np.load(p)
# res = npz['res'][()]
# alg_key = ['Attention_BS_Conv', 'Attention_BS_FC', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF']
# for k in alg_key:
#     band = res[k]['top_50_bands']
#     time = res[k]['time']
#     print('alg: %s\n time= %s\n band_index:%s ' % (k, time, band.tolist()))
# mobs_indian = [9, 12, 13, 15, 21, 36, 65, 66, 69, 80, 83, 90, 109, 116, 122, 134, 138, 157, 178, 179, 188]
# mobs_paviaU = [2, 4, 7, 10, 13, 17, 20, 31, 35, 52, 54, 59, 68, 77, 81, 82, 83, 85, 94, 95, 101]
# mobs_salinas = [9, 19, 20, 22, 34, 36, 38, 48, 59, 63, 95, 97, 105, 126, 138, 145, 152, 166, 172, 185, 200]

