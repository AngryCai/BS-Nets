# -*- coding: utf-8 -*-
"""
@ Description:
-------------

-------------
@ Time    : 2019/3/5 21:43
@ Author  : Yaoming Cai
@ FileName: demo-quantitative-evaluate-bandsubset-MSD.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
from sklearn.preprocessing import minmax_scale

from DeepLearning.AttentionBandSelection.classes.utility import cal_mean_spectral_divergence, cal_mean_spectral_angle
import matplotlib.pyplot as plt

from Toolbox.Preprocessing import Processor

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

processor = Processor()
img, gt = processor.prepare_data(img_path, gt_path)
# Img, Label = Img[:256, :, :], Label[:256, :]
n_row, n_column, n_band = img.shape
X_img = minmax_scale(img.reshape((n_row * n_column, n_band))).reshape((n_row, n_column, n_band))

p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_alg_bandindex_runtime-IndianPines.npz'
p_mobs = 'F:\Python\DeepLearning\AttentionBandSelection\\results\MOBS-indian-paviaU-salinas-top50band-index.npz'
p_opbs = 'F:\Python\DeepLearning\AttentionBandSelection\\results\OPBS-indian-paviaU-salinas-top50band-index.npz'

im_key = 'Indian'
a_mobs = np.load(p_mobs)
mobs_indx = a_mobs[im_key]
a_opbs = np.load(p_opbs)
opbs_indx = a_opbs[im_key]

a = np.load(p)
score = a['res'][()]
alg_key = ['Attention_BS_Conv', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
x_ = np.arange(3, 30, 2)
res_dict = {'Attention_BS_Conv':{'MSD':[], 'MSA':[]},
            'Attention_BS_FC':{'MSD':[], 'MSA':[]},
            'ISSC':{'MSD':[], 'MSA':[]},
            'SpaBS':{'MSD':[], 'MSA':[]},
            'MVPCA':{'MSD':[], 'MSA':[]},
            'SNMF':{'MSD':[], 'MSA':[]},
            'MOBS':{'MSD':[], 'MSA':[]},
            'OPBS':{'MSD':[], 'MSA':[]}}
for k in alg_key:
    print('alg:', k)
    for b_i in x_:
        if k == 'MOBS':
            band_indx = mobs_indx[b_i - 1]
        elif k == 'OPBS':
            band_indx = opbs_indx[:b_i]
        else:
            band_indx = score[k]['top_50_bands'][:b_i]
        x_b = X_img[:, :, band_indx]
        msd = cal_mean_spectral_divergence(x_b)
        msa = cal_mean_spectral_angle(x_b)
        res_dict[k]['MSD'].append(msd)
        res_dict[k]['MSA'].append(msa)
        print('\t\tmsd=%s, msa=%s' % (msd, msa))
np.savez('F:\Python\DeepLearning\AttentionBandSelection\\results\msd-msa-' + im_ + '-[3-30]-band.npz', res=res_dict)


"""
============================
 Plot MSD for each BS method
============================
"""
#
# import matplotlib.pyplot as plt
# import numpy as np
# FONTSIZE = 12
# LINEWIDTH = 1.8
# MARKERSIZE = 8
# CLASSIFIER = 'svm'  # 'knn'
# INDICATOR = 'oa'  # 'kappa'
# p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\msd-msa-Indian_pines_corrected-[3-30]-band.npz'
# npz = np.load(p)
# plot_key = ['BS-Net-Conv', 'BS-Net-FC', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
# marker = ['o', 'v', 's', 'd', 'H', '*', '^', 'D']
# alg_key = ['Attention_BS_Conv', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
# res = npz['res'][()]
#
# fig, ax = plt.subplots()
# x_ = np.arange(3, 30, 2)
# j = 0
# lines = []
# res_key = 'MSA'
# for k in alg_key:
#     s = res[k][res_key]
#     line, = ax.plot(x_, s, linestyle='-', marker=marker[j], markerfacecolor='None', linewidth=LINEWIDTH, markersize=MARKERSIZE, label=plot_key[j])
#     lines.append(line)
#     j += 1
#
# plt.legend(handles=lines, loc=1, prop={'size': FONTSIZE}, shadow=True)
# ymin, ymax = plt.ylim()
# plt.ylim(ymin-ymin/2, ymax)
# ax.set_xlabel('Number of selected bands', fontsize=FONTSIZE)
# # Make the y-axis label, ticks and tick labels match the line color.
# ax.set_ylabel(res_key, fontsize=FONTSIZE)
# ax.tick_params('y', labelsize=FONTSIZE)
# ax.tick_params('x', labelsize=FONTSIZE)
# plt.grid(True)
# plt.show()

