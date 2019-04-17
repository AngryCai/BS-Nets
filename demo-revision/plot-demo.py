"""
=====================
Plot loss-accuracy for MLP+Conv
=====================
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import minmax_scale
# root_res = 'F:\Python\DeepLearning\AttentionBandSelection\\results\\'
# # path = 'history-PaviaU-Conv-att-100epoch-5band-best.npz'
# path = root_res + '\\history-MLP-PaviaU-100epoch-5band.npz'
# npz_conv = np.load(path)
# score_conv = npz_conv['score']
# loss_conv = npz_conv['loss']
# w_conv = npz_conv['channel_weight']
# FONTSIZE = 12
# LINEWIDTH = 1.8
#
# oa = []
# EPOCH = len(score_conv)
# for i in range(EPOCH):
#     oa_ = score_conv[i]['svm']['oa'][0]
#     oa.append(oa_)
# x_ = range(EPOCH)
# fig, ax1 = plt.subplots()
# # line_1, = ax1.plot(x_, s1, linestyle='-', color='blue', marker='o', markerfacecolor='None', linewidth=LINEWIDTH, label='KNN')
# line_2, = ax1.plot(x_, oa, linestyle='-', color='blue', marker='s', markerfacecolor='None', linewidth=LINEWIDTH, label='SVM')
# ax1.set_xlabel('Epoch', fontsize=FONTSIZE)
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('OA (%)', color='blue', fontsize=FONTSIZE)
# ax1.tick_params('y', colors='blue', labelsize=FONTSIZE)
# ax1.tick_params('x', labelsize=FONTSIZE)
# # ax1.legend(loc=5)
#
# ax2 = ax1.twinx()
# line_3, = ax2.plot(x_, loss_conv, linestyle='-', color='orangered', linewidth=LINEWIDTH, label='MSE')
# ax2.set_ylabel('Loss', color='orangered', fontsize=FONTSIZE)
# ax2.tick_params('y', colors='orangered', labelsize=FONTSIZE)
# # ax2.legend(loc=5)
# # plt.legend(handles=[line_2, line_3], loc=5, prop={'size': 12}, shadow=True)
# fig.tight_layout()
# plt.grid(True)
# plt.show()



"""
=====================
Plot running time: MLP v.s Conv
=====================
"""
import numpy as np
import matplotlib.pyplot as plt
FONTSIZE = 12
LINEWIDTH = 1.8

path_mlp = 'F:\Python\DeepLearning\AttentionBandSelection\\results\\run-time-epoch-MLP.npz'
path_conv = 'F:\Python\DeepLearning\AttentionBandSelection\\results\\run-time-epoch-conv.npz'
time_mlp = np.load(path_mlp)['run_time']
time_conv = np.load(path_conv)['run_time']
EPOCH = len(time_mlp)
x_ = range(EPOCH)
fig, ax = plt.subplots()
line_mlp, = ax.plot(x_, time_mlp, linestyle='-', linewidth=LINEWIDTH, label='BS-Net-FC')
line_conv, = ax.plot(x_, time_conv, linestyle='--', linewidth=LINEWIDTH, label='BS-Net-Conv')
ax.set_xlabel('Epoch', fontsize=FONTSIZE)
# Make the y-axis label, ticks and tick labels match the line color.
ax.set_ylabel('Training time (s)', fontsize=FONTSIZE)
ax.tick_params('y', labelsize=FONTSIZE)
ax.tick_params('x', labelsize=FONTSIZE)
plt.legend(handles=[line_mlp, line_conv], loc='best', prop={'size': FONTSIZE}, shadow=True)
fig.tight_layout()
plt.grid(True)
plt.show()


"""
=====================
Plot Band Index for all BS methods
=====================
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from Toolbox.Preprocessing import Processor
import numpy as np

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
FONTSIZE = 12
LINEWIDTH = 1.2
MARKERSIZE = 16
plot_key = ['BS-Net-FC', 'BS-Net-Conv', 'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
marker = ['o', 'v', 's', 'd', 'H', '*', '^', 'D']

x = np.arange(n_band)

index = [
[165, 38, 51, 65, 12, 100, 0, 71, 5, 60, 88, 26, 164, 75, 74],
[46,33,140,161,80,35,178,44,126,36,138,71,180,66,192],
[171,130,67,85,182,183,47,143,138,90,139,141,25,142,21],
[7, 96, 52, 171, 53, 3, 76, 75, 74, 95, 77, 73, 78, 54, 81],
[167,74,168,0,147,165,161,162,152,19,160,119,164,159,157],
[23,197,198,94,76,2,87,105,143,145,11,84,132,108,28],
[5,6,19,24,45,48,105,114,129,142,144,160,168,172,181],
[28, 41, 60, 0, 74, 34, 88, 19, 17, 33, 56, 87, 22, 31, 73]

]

fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
# fig = plt.figure(figsize=(15, 5))
# Plot all fill styles.
for y, k in enumerate(plot_key):
    line, = ax[0].plot(index[y], y * np.ones(len(index[0])), linestyle='-', marker='.',
                    markerfacecolor='None', markersize=MARKERSIZE, linewidth=.2)
# # uniformly settings
ax[0].set_yticks(np.arange(len(plot_key)))
ax[0].set_yticklabels(plot_key, fontsize=FONTSIZE)
# ax[0].set_xlabel('Spectral band', fontsize=FONTSIZE)
ax[0].tick_params('y', labelsize=FONTSIZE)
ax[0].tick_params('x', labelsize=FONTSIZE)
ax[0].grid(axis='x')

"""
=====================
cal. and plot entropy
=====================
"""
X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
hist = []
N = n_row * n_column
for i in range(n_band):
    hist_, edge_ = np.histogram(X_img[:, :, i], 256)
    hist.append(hist_ / N)
hist = np.asarray(hist)
entropy_ = entropy(hist.transpose())
# fig, ax = plt.subplots(figsize=(15, 5))
FONTSIZE = 12
LINEWIDTH = 1.8
ax[1].plot(entropy_, linewidth=1.8)
# plt.xticks(np.arange(0, n_band)[band_indx], band_indx)
ax[1].set_xlabel('Spectral band', fontsize=FONTSIZE)
ax[1].set_ylabel('Value of entropy', fontsize=FONTSIZE)
ax[1].tick_params('y', labelsize=FONTSIZE)
ax[1].tick_params('x', labelsize=FONTSIZE)
ax[1].grid(axis='x')
fig.tight_layout()
plt.show()
