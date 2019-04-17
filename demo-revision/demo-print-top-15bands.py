import numpy as np
from sklearn.preprocessing import minmax_scale
from Toolbox.Preprocessing import Processor


p = 'F:\Python\DeepLearning\AttentionBandSelection\\results\diff_alg_bandindex_runtime-Salinas.npz'
p_mobs = 'F:\Python\DeepLearning\AttentionBandSelection\\results\MOBS-indian-paviaU-salinas-top50band-index.npz'
p_opbs = 'F:\Python\DeepLearning\AttentionBandSelection\\results\OPBS-indian-paviaU-salinas-top50band-index.npz'
a_mobs = np.load(p_mobs)
a_opbs = np.load(p_opbs)
im = 'PaviaU'
print('data set:', im)
mobs_indx = a_mobs[im]
opbs_indx = a_opbs[im]

a = np.load(p)
score = a['res'][()]

alg_key = ['Attention_BS_Conv', 'Attention_BS_FC',  'ISSC', 'SpaBS', 'MVPCA', 'SNMF', 'MOBS', 'OPBS']
n_band = 50
for k in alg_key:
    print('===============================')
    print('alg:', k)
    if k == 'MOBS':
        band_indx = mobs_indx[n_band - 1]
    elif k=='OPBS':
        band_indx = opbs_indx[:n_band]
    else:
        band_indx = score[k]['top_50_bands'][:n_band]
    print(list(band_indx))