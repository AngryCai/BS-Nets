# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/11/12 14:26
@ Author  : Yaoming Cai
@ FileName: Attention_CNN_loss_acc_temp.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import sys

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.append('/home/caiyaom/python_codes/')

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import maxabs_scale
from tensorflow.contrib.layers import *
from Toolbox.Helper import Dataset
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from utility import eval_band_cv

class Attention_CNN:

    def __init__(self, lr, batch_size, epoch, n_selected_band):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_selected_band = n_selected_band

    def net(self, x_input, is_training=True):
        n_channel = x_input.get_shape().as_list()[-1]
        input_norm = tf.layers.batch_normalization(x_input, training=is_training, name='input_norm')

        # # attention net
        conv_att_1 = tf.layers.conv2d(input_norm, 64, (6, 6), strides=(1, 1), padding='valid',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='attention-1')
        bn_att_1 = tf.nn.relu(tf.layers.batch_normalization(conv_att_1, training=is_training), name='BN-att-1')

        # conv_att_2 = tf.layers.conv2d(bn_att_1, 32, (3, 3), strides=(1, 1), padding='valid',
        #                               kernel_initializer=tf.random_uniform_initializer, name='attention-2')
        # bn_att_2 = tf.nn.relu(tf.layers.batch_normalization(conv_att_2, training=is_training), name='BN-att-2')

        global_pool = tf.reduce_mean(bn_att_1, axis=[1, 2], name='global_pooling')
        bottleneck = tf.layers.dense(global_pool, 128, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='bottleneck')
        channel_weight = tf.layers.dense(bottleneck, n_channel, activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activity_regularizer=l1_regularizer(0.01), name='channel_weight')
        channel_weight_ = tf.reshape(channel_weight, [-1, 1, 1, n_channel], name='weight_reshape')
        reweight_out = channel_weight_ * input_norm

        # # conv net
        conv_1 = tf.layers.conv2d(reweight_out, 128, (3, 3), strides=(1, 1), padding='valid',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv-1')
        batch_norm_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training), name='BN-1')

        conv_2 = tf.layers.conv2d(batch_norm_1, 64, (3, 3), strides=(1, 1), padding='valid',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv-2')
        batch_norm_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training), name='BN-2')

        # conv_3 = tf.layers.conv2d(batch_norm_2, 32, (3, 3), strides=(1, 1), padding='valid',
        #                           kernel_initializer=tf.random_uniform_initializer, name='conv-code')
        # batch_norm_3 = tf.nn.relu(tf.layers.batch_normalization(conv_3, training=is_training), name='BN-code')

        conv_tr_1 = tf.layers.conv2d_transpose(batch_norm_2, 64, (3, 3), strides=(1, 1), padding='valid',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='tran-conv-1')
        conv_tr_bn_1 = tf.nn.relu(tf.layers.batch_normalization(conv_tr_1, training=is_training), name='BN-3')

        conv_tr_2 = tf.layers.conv2d_transpose(conv_tr_bn_1, 128, (3, 3), strides=(1, 1), padding='valid',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='tran-conv-2')
        conv_tr_bn_2 = tf.nn.relu(tf.layers.batch_normalization(conv_tr_2, training=is_training), name='BN-4')

        output = tf.layers.conv2d(conv_tr_bn_2, n_channel, (1, 1), strides=(1, 1), padding='same',
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='recons')
        return channel_weight, output

    def fit(self, X, X_2, img=None, gt=None):
        n_sam, n_row, n_clm, n_channel = X.shape
        self.x_placehoder = tf.placeholder(shape=(None, None, None, n_channel), dtype=tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        # self.is_fine_tuning = tf.placeholder(tf.bool)
        channel_weight, output = self.net(self.x_placehoder, is_training=self.is_training)
        tf.summary.histogram('channel_weight', channel_weight)
        self.loss_recons = tf.losses.mean_squared_error(self.x_placehoder, output) + \
                           tf.losses.get_regularization_loss()
        tf.summary.scalar('loss', self.loss_recons)
        tf.summary.merge_all()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_recons)
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)  # allocate gpu memory according to model's need
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', sess.graph)
        dataset_1 = Dataset(X, X)
        dataset_2 = Dataset(X_2, X_2)
        loss_history = []
        score_list = []
        channel_weight_list = []
        for i_epoch in range(self.epoch):
            for batch_i in range(n_sam // self.batch_size):
                x_batch, y_batch = dataset_1.next_batch(self.batch_size, shuffle=True)
                train_op.run(feed_dict={self.x_placehoder: x_batch, self.is_training: True})
            for batch_i in range(X_2.shape[0] // self.batch_size):
                x_batch, y_batch = dataset_2.next_batch(self.batch_size, shuffle=True)
                train_op.run(feed_dict={self.x_placehoder: x_batch, self.is_training: True})

            loss_reocns_1, channel_weight_1, summury_1 = sess.run([self.loss_recons, channel_weight, merged],
                                                                  feed_dict={self.x_placehoder: X,
                                                                             self.is_training: False})
            loss_reocns_, channel_weight_2, summury = sess.run([self.loss_recons, channel_weight, merged],
                                                               feed_dict={self.x_placehoder: X_2,
                                                                          self.is_training: False})
            print('epoch %s ==> loss=%s' % (i_epoch, loss_reocns_))
            loss_history.append(loss_reocns_)
            writer.add_summary(summury, i_epoch)
            # if i_epoch >= 2:
            weight = np.vstack((channel_weight_1, channel_weight_2))
            channel_weight_list.append(weight)
            if img is not None:
                # score = self.eval_band(img, gt, channel_weight_, train_inx, test_idx, self.n_selected_band)
                # score = self.eval_band_cv(img, gt, weight, self.n_selected_band, times=2)
                mean_weight = np.mean(weight, axis=0)
                band_indx = np.argsort(mean_weight)[::-1][:self.n_selected_band]
                print('=============================')
                print('SELECTED BAND: ', band_indx)
                print('=============================')
                x_new = img[:, :, band_indx]
                n_row, n_clm, n_band = x_new.shape
                img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
                p = Processor()
                img_correct, gt_correct = p.get_correct(img_, gt)
                score = eval_band_cv(img_correct, gt_correct, times=30, test_size=0.95)
                print('acc=', score)
                score_list.append(score)
            if i_epoch % 10 == 0:
                np.savez('history.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
        np.savez('history.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
        saver.save(sess, './IndianPine-model.ckpt')

    # def eval_band_cv(self, img, gt, channel_weight, num_selected=10, times=2):
    #     """
    #     :param X:
    #     :param y:
    #     :param times: n times k-fold cv
    #     :return:  knn/svm/elm=>(OA+std, Kappa+std)
    #     """
    #     mean_weight = np.mean(channel_weight, axis=0)
    #     band_indx = np.argsort(mean_weight)[-num_selected:]
    #     print('=============================')
    #     print('SELECTED BAND: ', band_indx)
    #     print('=============================')
    #     x_new = img[:, :, band_indx]
    #     n_row, n_clm, n_band = x_new.shape
    #     img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
    #     p = Processor()
    #     img_correct, gt_correct = p.get_correct(img_, gt)
    #
    #     estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
    #     estimator_pre, y_test_all = [[], []], []
    #     for i in range(times):  # repeat N times K-fold CV
    #         skf = StratifiedKFold(n_splits=10, shuffle=True)
    #         for test_index, train_index in skf.split(img_correct, gt_correct):
    #             X_train, X_test = img_correct[train_index], img_correct[test_index]
    #             y_train, y_test = gt_correct[train_index], gt_correct[test_index]
    #             y_test_all.append(y_test)
    #             for c in range(len(estimator)):
    #                 estimator[c].fit(X_train, y_train)
    #                 y_pre = estimator[c].predict(X_test)
    #                 estimator_pre[c].append(y_pre)
    #     clf = ['knn', 'svm', 'dnn']
    #     score = []
    #     for z in range(len(estimator)):
    #         ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=clf[z] + 'score.npz',
    #                                                   verbose=True)
    #         score.append([oa, kappa])
    #     return score


'''
===================================
                Test
===================================
'''
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
import numpy as np
from skimage.util.shape import view_as_windows
import time
if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    # root = '/home/caiyaom/HSI_Files/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
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
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_block = view_as_windows(X_img, (16, 16, n_band), step=6)
    img_block_2 = view_as_windows(X_img, (32, 32, n_band), step=6)
    img_block = img_block.reshape((-1, img_block.shape[-3], img_block.shape[-2], img_block.shape[-1]))
    img_block_2 = img_block_2.reshape((-1, img_block_2.shape[-3], img_block_2.shape[-2], img_block_2.shape[-1]))

    print('img block shape: ', img_block.shape, img_block_2.shape)

    # img_blocks_nonzero, gt_blocks_nonzero = p.divide_img_blocks(X_img, gt, block_size=(8, 8))
    # img_blocks_nonzero = np.transpose(img_blocks_nonzero, [0, 2, 3, 1]).astype('float32')
    # print(img_blocks_nonzero.shape)

    LR, BATCH_SIZE, EPOCH = 0.0001, 32, 100
    N_BAND = 5
    time_start = time.clock()
    acnn = Attention_CNN(LR, BATCH_SIZE, EPOCH, N_BAND)
    acnn.fit(img_block, img_block_2)
    run_time = round(time.clock() - time_start, 3)
    print('running time=', run_time)


"""
============================
PLOT WEIGHT MATRIX
============================
"""
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import minmax_scale
# from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
# path= 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-Indian-Conv-att-100epoch-5band-best.npz'
# npz = np.load(path)
# # s = npz['score']
# loss_ = npz['loss']
# w = npz['channel_weight']
# FONTSIZE = 12
# w_mean = np.mean(w, axis=1)
# scale_w = minmax_scale(w_mean, axis=1)
# fig, ax = plt.subplots()
# img = ax.imshow(scale_w, cmap='jet')
# ax.set_ylabel('Epoch', fontsize=FONTSIZE)
# ax.set_xlabel('Spectral band', fontsize=FONTSIZE)
#
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.2)
# cbar = fig.colorbar(img, cax=cax)
# cbar.ax.tick_params(labelsize=FONTSIZE)
# ax.tick_params('y', labelsize=FONTSIZE)
# ax.tick_params('x', labelsize=FONTSIZE)


"""
============================
PLOT LOSS-ACCURACY NEW
============================
"""
# import numpy as np
# import matplotlib.pyplot as plt
# path = 'C:\\Users\\07\Desktop\putty\history.npz'
# npz = np.load(path)
# score = npz['score']
# loss_ = npz['loss']
# w = npz['channel_weight']
# FONTSIZE = 12
# LINEWIDTH = 1.8
#
# oa = []
# EPOCH = len(score)
# for i in range(EPOCH):
#     oa_ = score[i]['svm']['oa'][0]
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
# line_3, = ax2.plot(x_, loss_, linestyle='-',  color='orangered', linewidth=LINEWIDTH, label='MSE')
# ax2.set_ylabel('Loss', color='orangered', fontsize=FONTSIZE)
# ax2.tick_params('y', colors='orangered', labelsize=FONTSIZE)
# # ax2.legend(loc=5)
# # plt.legend(handles=[line_2, line_3], loc=5, prop={'size': 12}, shadow=True)
# fig.tight_layout()
# plt.grid(True)
# plt.show()


"""
=====================
show band image
=====================
"""
# import matplotlib.pyplot as plt
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
# img_path = root + im_ + '.mat'
# gt_path = root + gt_ + '.mat'
# print(img_path)
# p = Processor()
# img, gt = p.prepare_data(img_path, gt_path)
# n_row, n_column, n_band = img.shape
# X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
# band_indx = [93, 123, 68, 32, 76] # indina: [46, 33, 140, 161, 80] # PaviaU: [90, 42, 16, 48, 71]  # Salinas: [93, 123, 68, 32, 76]
# fig, axes = plt.subplots(1, 5, figsize=(12, 3))
# fig.subplots_adjust(hspace=0, wspace=0)
# i = 0
# for a in axes:
#     a.imshow(img[:, :, band_indx[i]])
#     a.set_title('Band %s' % band_indx[i])
#     # a.set_axis_off()
#     i += 1
# fig.tight_layout()
# plt.show()
