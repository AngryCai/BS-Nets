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
        global_pool = tf.reduce_mean(input_norm, axis=[1, 2], name='global_pooling')
        bottleneck = tf.layers.dense(global_pool, 128, activation=tf.nn.relu, name='bottleneck')
        channel_weight = tf.layers.dense(bottleneck, n_channel, activation=tf.nn.sigmoid,
                                         activity_regularizer=l1_regularizer(1e8), name='channel_weight')
        channel_weight_ = tf.reshape(channel_weight, [-1, 1, 1, n_channel], name='weight_reshape')
        reweight_out = channel_weight_ * input_norm

        # # conv net
        conv_1 = tf.layers.conv2d(reweight_out, 128, (3, 3), strides=(1, 1), padding='valid',
                                  kernel_initializer=tf.random_uniform_initializer, name='conv-1')
        batch_norm_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training), name='BN-1')

        conv_2 = tf.layers.conv2d(batch_norm_1, 64, (3, 3), strides=(1, 1), padding='valid',
                                  kernel_initializer=tf.random_uniform_initializer, name='conv-2')
        batch_norm_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training), name='BN-2')

        conv_tr_1 = tf.layers.conv2d_transpose(batch_norm_2, 64, (3, 3), strides=(1, 1), padding='valid',
                                               kernel_initializer=tf.random_uniform_initializer, name='tran-conv-1')
        conv_tr_bn_1 = tf.nn.relu(tf.layers.batch_normalization(conv_tr_1, training=is_training), name='BN-3')

        conv_tr_2 = tf.layers.conv2d_transpose(conv_tr_bn_1, 128, (3, 3), strides=(1, 1), padding='valid',
                                               kernel_initializer=tf.random_uniform_initializer, name='tran-conv-2')
        conv_tr_bn_2 = tf.nn.relu(tf.layers.batch_normalization(conv_tr_2, training=is_training), name='BN-4')

        output = tf.layers.conv2d(conv_tr_bn_2, n_channel, (1, 1), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.random_uniform_initializer, name='recons')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))

        return channel_weight, output

    def fit(self, X, img=None, gt=None):
        if img is not None:
            p = Processor()
            img_correct, gt_correct = p.get_correct(img, gt)
            train_inx, test_idx = p.get_tr_ts_index_num(gt_correct, n_labeled=20)
        n_sam, n_row, n_clm, n_channel = X.shape
        self.x_placehoder = tf.placeholder(shape=(None, n_row, n_clm, n_channel), dtype=tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        # self.is_fine_tuning = tf.placeholder(tf.bool)
        channel_weight, output = self.net(self.x_placehoder, is_training=self.is_training)
        tf.summary.histogram('channel_weight', channel_weight)
        self.loss_recons = tf.losses.mean_squared_error(self.x_placehoder, output)
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
        dataset = Dataset(X, X)
        loss_history = []
        score_list = []
        for i_epoch in range(self.epoch):
            # if i_epoch > self.epoch // :
            for batch_i in range(n_sam // self.batch_size):
                x_batch, y_batch = dataset.next_batch(self.batch_size, shuffle=True)
                train_op.run(feed_dict={self.x_placehoder: x_batch, self.is_training: True})
            # else:
            #     train_op.run(feed_dict={self.x_placehoder: X, self.is_training: True})
            loss_reocns_, channel_weight_, summury = sess.run([self.loss_recons, channel_weight, merged],
                                                     feed_dict={self.x_placehoder: X, self.is_training: False})
            print('epoch %s ==> loss=%s' % (i_epoch, loss_reocns_))
            loss_history.append(loss_reocns_)
            writer.add_summary(summury, i_epoch)
            if img is not None:
                # score = self.eval_band(img, gt, channel_weight_, train_inx, test_idx, self.n_selected_band)
                score = self.eval_band_cv(img, gt, channel_weight_, self.n_selected_band, times=2)
                print('acc=', score)
                score_list.append(score)
        np.savez('history-1.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_)
        saver.save(sess, './IndianPine-model.ckpt')

    def eval_band(self, img, gt, channel_weight, train_inx, test_idx, num_selected=10):
        """
        :param new_img:
        :param gt:
        :param train_inx:
        :param test_idx:
        :return:
        """
        mean_weight = np.mean(channel_weight, axis=0)
        band_indx = np.argsort(mean_weight)[-num_selected:]
        x_new = img[:, :, band_indx]
        n_row, n_clm, n_band = x_new.shape
        img_ = maxabs_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
        p = Processor()
        img_correct, gt_correct = p.get_correct(img_, gt)
        # train_inx, test_idx = p.get_tr_ts_index_num(y, n_labeled=50)
        # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
        X_train, X_test, y_train, y_test = img_correct[train_inx], img_correct[test_idx], \
                                           gt_correct[train_inx], gt_correct[test_idx]
        classifier = SVC(C=1e6, kernel='rbf', gamma=1.)  # KNN(n_neighbors=3)
        classifier.fit(X_train, y_train)
        y_pre = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pre)
        return score

    def eval_band_cv(self, img, gt, channel_weight, num_selected=10, times=2):
        """
        :param X:
        :param y:
        :param times: n times k-fold cv
        :return:  knn/svm/elm=>(OA+std, Kappa+std)
        """
        mean_weight = np.mean(channel_weight, axis=0)
        band_indx = np.argsort(mean_weight)[-num_selected:]
        x_new = img[:, :, band_indx]
        n_row, n_clm, n_band = x_new.shape
        img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
        p = Processor()
        img_correct, gt_correct = p.get_correct(img_, gt)

        estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
        estimator_pre, y_test_all = [[], []], []
        for i in range(times):  # repeat N times K-fold CV
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            for test_index, train_index in skf.split(img_correct, gt_correct):
                X_train, X_test = img_correct[train_index], img_correct[test_index]
                y_train, y_test = gt_correct[train_index], gt_correct[test_index]
                y_test_all.append(y_test)
                for c in range(len(estimator)):
                    estimator[c].fit(X_train, y_train)
                    y_pre = estimator[c].predict(X_test)
                    estimator_pre[c].append(y_pre)
        clf = ['knn', 'svm', 'dnn']
        score = []
        for z in range(len(estimator)):
            ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=clf[z] + 'score.npz',
                                                      verbose=True)
            score.append([oa, kappa])
        return score


'''
===================================
                Test
===================================
'''
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
import numpy as np
from skimage.util.shape import view_as_windows

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
    img_block = view_as_windows(X_img, (8, 8, n_band), step=2)
    img_block = img_block.reshape((-1, img_block.shape[-3], img_block.shape[-2], img_block.shape[-1]))
    print('img block shape: ', img_block.shape)

    # img_blocks_nonzero, gt_blocks_nonzero = p.divide_img_blocks(X_img, gt, block_size=(8, 8))
    # img_blocks_nonzero = np.transpose(img_blocks_nonzero, [0, 2, 3, 1]).astype('float32')
    # print(img_blocks_nonzero.shape)

    LR, BATCH_SIZE, EPOCH = 0.001, 64, 50
    N_BAND = 3
    acnn = Attention_CNN(LR, BATCH_SIZE, EPOCH, N_BAND)
    acnn.fit(img_block, X_img, gt)

"""
============================
PLOT LOSS-ACCURACY 
============================
"""
# import numpy as np
# import matplotlib.pyplot as plt
# path = 'C:\\Users\\07\Desktop\putty\history-1.npz'
# # path= 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-SalinasA.npz'
# path = 'F:\Python\DeepLearning\AttentionBandSelection\classes\history-1.npz'
# npz = np.load(path)
# # s = npz['score']
# s1 = npz['score'][:, 0, 0, 0]
# s2 = npz['score'][:, 1, 0, 0]
# loss_ = npz['loss']
# w = npz['channel_weight']
# x_ = range(len(s1))
# FONTSIZE = 12
# LINEWIDTH = 1.8
#
# fig, ax1 = plt.subplots()
# line_1, = ax1.plot(x_, s1, linestyle='-', color='blue', marker='o', markerfacecolor='None', linewidth=LINEWIDTH, label='KNN')
# line_2, = ax1.plot(x_, s2, linestyle='-', color='blue', marker='s', markerfacecolor='None', linewidth=LINEWIDTH, label='SVM')
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
# plt.legend(handles=[line_1, line_2, line_3], loc=5, prop={'size': 12}, shadow=True)
# fig.tight_layout()
# plt.grid(True)
# plt.show()
tf.layers.Conv3D