# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/12/6 20:51
@ Author  : Yaoming Cai
@ FileName: demo_varying_band_varying_epoch.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
from DeepLearning.AttentionBandSelection.classes.utility import eval_band_cv
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from skimage.util.shape import view_as_windows

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


# path_weight = 'C:\\Users\\07\Desktop\putty\history-conv-att-PaviaU-100epoch-20band.npz'
path_weight = 'C:\\Users\\07\Desktop\putty\history.npz'
npz = np.load(path_weight)

channel_weight = npz['channel_weight']

score = []
for i in range(5, 6, 5):
    s = []
    for j in range(channel_weight.shape[0]):
        print('# band:', i, 'epoch:', j)
        mean_weight = np.mean(channel_weight[j], axis=0)
        band_indx = np.argsort(mean_weight)[-i:]
        print('=============================')
        print('SELECTED BAND: ', band_indx)
        print('=============================')
        x_new = img[:, :, band_indx]
        n_row, n_clm, n_band = x_new.shape
        img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
        p = Processor()
        img_correct, gt_correct = p.get_correct(img_, gt)
        score_ = eval_band_cv(img_correct, gt_correct, times=10, test_size=0.95)
        print('acc=', score_)
        s.append(score_)
    score.append(s)

np.savez('F:\Python\DeepLearning\AttentionBandSelection\\results\\paviaU-5band-200epoch.npz', score=score)


'''
=============================
PLOT 
=============================
'''
# import numpy as np
# import matplotlib.pyplot as plt
# p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\\varying_band_varying_epoch.npz'
# npz = np.load(p)
# score = npz['score']
# # ss = [score[:100], score[100:]]
# oa = []
# CASE = 5
# EPOCH = 100
# for i in range(CASE):
#     s = []
#     for j in range(EPOCH):
#         oa_ = score[i][j]['svm']['oa'][0]
#         s.append(oa_)
#     oa.append(s)
#
# x = np.arange(5, 31, 5)
# for i in range(CASE):
#     plt.plot(np.asarray(oa)[i], label=str(x[i]))
# plt.legend()





