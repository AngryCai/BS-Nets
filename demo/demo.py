from DeepLearning.AttentionBandSelection.classes.SpaBS import SpaBS
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
import numpy as np
from DeepLearning.AttentionBandSelection.classes.utility import eval_band, eval_band_cv
from DeepLearning.AttentionBandSelection.classes.ISSC import ISSC_HSI
from DeepLearning.AttentionBandSelection.classes.MVPCA_BS import MVPCA


if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_correct, gt_correct = p.get_correct(X_img, gt)
    gt_correct = p.standardize_label(gt_correct)
    X_img_2D = X_img.reshape(n_row * n_column, n_band)
    n_selected_band = 5

    algorithm = [
                 SpaBS(n_selected_band),
                 ISSC_HSI(n_selected_band, coef_=1.e-4),
                 MVPCA(n_selected_band)
    ]
    alg_key = ['SpaBS', 'ISSC']
    for i in range(algorithm.__len__()):
        X_new = algorithm[i].predict(X_img_2D)
        X_new_3D = X_new.reshape((n_row, n_column, n_selected_band))
        img_correct, gt_correct = p.get_correct(X_img, gt)
        gt_correct = p.standardize_label(gt_correct)
        score = eval_band_cv(img_correct, gt_correct)
        print('%s:  %s' % (alg_key[i], score))
        print('-------------------------------------------')
