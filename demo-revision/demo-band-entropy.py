'''
===================================
    Demo: analyze band entropy
===================================
'''

from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from Toolbox.Preprocessing import Processor
import numpy as np

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
X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
hist = []
N = n_row * n_column
for i in range(n_band):
    hist_, edge_ = np.histogram(X_img[:, :, i], 256)
    hist.append(hist_ / N)
hist = np.asarray(hist)
entropy_ = entropy(hist.transpose())
# np.savez('F:\Python\DeepLearning\AttentionBandSelection\\results\\entropy-histogram-' + im_ + '.npz', entropy=entropy_)


"""
=====================
show spectral band with best bands
=====================
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace=0, wspace=0)
FONTSIZE = 12
axes.plot(entropy_, linewidth=1.8)
# plt.xticks(np.arange(0, n_band)[band_indx], band_indx)
axes.set_xlabel('Spectral band', fontsize=FONTSIZE)
axes.set_ylabel('Value of entropy', fontsize=FONTSIZE)
axes.tick_params('y', labelsize=FONTSIZE)
axes.tick_params('x', labelsize=FONTSIZE)
# # #  Conv:
#  Indian: [46, 33, 140, 161, 80, 35, 178, 44, 126, 36, 138, 71, 180, 66, 192]
#  PaviaU:[90, 42, 16, 48, 71, 3, 78, 38, 80, 53, 7, 31, 4, 99, 98]
# Salinas: [93, 123, 68, 32, 106, 76, 90, 7, 173, 67, 39, 44, 195, 168, 98]
band_indx = [93, 123, 68, 32, 116, 76, 90, 7, 173, 67, 39, 44, 195, 168, 98]
for j in band_indx:
    plt.axvline(j, linestyle='--', color='r', linewidth=2, alpha=0.5)


# # # FC:
# indian: [165, 38, 51, 65, 12, 100, 0, 71, 5, 60, 88, 26, 164, 75, 74]
# paviaU:[38, 78, 17, 20, 85, 98, 65, 81, 79, 90, 95, 74, 66, 62, 92]
# salinas: [13, 93, 47, 128, 55, 163, 20, 65, 126, 39, 83, 34, 188, 69, 90]
band_indx = [13, 93, 47, 128, 55, 163, 20, 65, 126, 39, 83, 34, 188, 69, 90]
for j in band_indx:
    plt.axvline(j, linestyle='--', color='g', linewidth=2, alpha=0.5)
plt.show()
