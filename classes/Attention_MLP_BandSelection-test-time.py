# -*- coding: utf-8 -*-
"""
@ Description: 
-------------
Deep Fullly Connected Nets (aka. MLP) with Band Attention Module
-------------
@ Time    : 2019/2/28 15:32
@ Author  : Yaoming Cai
@ FileName: Attention_MLP_BandSelection.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import time

import numpy as np
import sys

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.append('/home/caiyaom/python_codes/')

import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.layers import *
from Toolbox.Helper import Dataset
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from utility import eval_band_cv
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale


class Attention_MLP:

    def __init__(self, lr, batch_size, epoch, n_selected_band):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_selected_band = n_selected_band

    def net(self, x_input, is_training=True):
        """
        :param x_input: with shape of (N, n_bands)
        :param is_training:
        :return:
        """
        n_channel = x_input.get_shape().as_list()[-1]
        input_norm = tf.layers.batch_normalization(x_input, training=is_training, name='input_norm')

        # # attention module
        dense_att_1 = tf.layers.dense(input_norm, 64, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='attention-1')
        bn_att_1 = tf.nn.relu(tf.layers.batch_normalization(dense_att_1, training=is_training), name='BN-att-1')
        bottleneck = tf.layers.dense(bn_att_1, 128, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='bottleneck')
        channel_weight = tf.layers.dense(bottleneck, n_channel, activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activity_regularizer=l1_regularizer(0.01), name='channel_weight')
        channel_weight_ = tf.reshape(channel_weight, [-1, n_channel], name='weight_reshape')
        reweight_out = channel_weight_ * input_norm

        # # conv net
        # Encoder
        fcn_1 = tf.layers.dense(reweight_out, 64, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='fcn-1')
        batch_norm_1 = tf.nn.relu(tf.layers.batch_normalization(fcn_1, training=is_training), name='BN-1')

        fcn_2 = tf.layers.dense(batch_norm_1, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='fcn-2')
        batch_norm_2 = tf.nn.relu(tf.layers.batch_normalization(fcn_2, training=is_training), name='BN-2')

        # Decoder
        fcn_3 = tf.layers.dense(batch_norm_2, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='fcn-3')
        batch_norm_3 = tf.nn.relu(tf.layers.batch_normalization(fcn_3, training=is_training), name='BN-3')

        fcn_4 = tf.layers.dense(batch_norm_3, n_channel, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='fcn-4')
        output = tf.nn.sigmoid(tf.layers.batch_normalization(fcn_4, training=is_training), name='recons')

        return channel_weight, output

    def fit(self, X, img=None, gt=None):
        n_sam, n_channel = X.shape
        self.x_placehoder = tf.placeholder(shape=(None, n_channel), dtype=tf.float32)
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
        dataset = Dataset(X, X)
        loss_history = []
        score_list = []
        channel_weight_list = []
        time_start = time.clock()
        time_list = []
        for i_epoch in range(self.epoch):
            for batch_i in range(n_sam // self.batch_size):
                x_batch, y_batch = dataset.next_batch(self.batch_size, shuffle=True)
                _, loss_ = sess.run([train_op, self.loss_recons], feed_dict={self.x_placehoder: x_batch, self.is_training: True})
                # train_op.run(feed_dict={self.x_placehoder: x_batch, self.is_training: True})
                print('%s epoch, %s batch: %s' % (i_epoch, batch_i, loss_))
            run_time = round(time.clock() - time_start, 3)
            time_list.append(run_time)
        np.savez('run-time-epoch.npz', run_time=np.asarray(time_list))


'''
===================================
        Demo: train model
===================================
'''
if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    # root = '/home/caiyaom/HSI_Files/'
    # root = '/home/caiyaom/Research/HSI_Files/'
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
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))

    X_train = np.reshape(X_img, (n_row * n_column, n_band))
    print('training img shape: ', X_train.shape)

    LR, BATCH_SIZE, EPOCH = 0.00002, 64, 100
    N_BAND = 5
    time_start = time.clock()
    acnn = Attention_MLP(LR, BATCH_SIZE, EPOCH, N_BAND)
    acnn.fit(X_train)
    run_time = round(time.clock() - time_start, 3)
    print('running time=', run_time)


