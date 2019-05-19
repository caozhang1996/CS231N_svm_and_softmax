#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:00:11 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math 
import time
import numpy as np
import matplotlib.pyplot as plt

from classifier.linear_svm import *
from classifier.linear_classifier import *
import data_utils


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    """
    root_dir = '../dataset/cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = data_utils.load_CIFAR10(root_dir)
    
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]
    
    #reshape and subtract the mean
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    mean_image = np.mean(X_train, axis=0)             # 这一步是将数据数据零均值化
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev


def compare_difference():
    """
    """
    W = np.random.randn(3072, 10) * 1e-4
    B = np.zeros(10) 
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()
    
    tic = time.time()      # 函数执行前的时间
    loss_navie, grad_naive = svm_loss_navie(W, B, X_dev, Y_dev, 0.000005)
    toc = time.time()      # 函数执行完毕的时间
    print ('naive loss: %e compute in %fs' % (loss_navie, toc - tic))
    
    tic = time.time()      # 函数执行前的时间
    loss_vectorized, grad_vectorized = svm_loss_vectorized(W, B, X_dev, Y_dev, 0.000005)
    toc = time.time()      # 函数执行完毕的时间
    print ('vectorized loss: %e compute in %fs' % (loss_vectorized, toc - tic))
    
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized)
    print ('Loss difference: %f' % np.abs(loss_navie - loss_vectorized))
    print ('Gradient difference: %f' % grad_difference)


def svm_train():
    """
    """
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()
    svm = LinearSVM()
    tic = time.time()
    loss_history = svm.train(X_train, Y_train, learning_rate=1e-7, reg=2.5e4,
                             num_iters=2500, verbose=True)
    toc = time.time()
    print ('That took %fs: '% (toc - tic))
    
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
    
    # 用学习到的参数去评估训练集,验证集和测试集的表现
    y_train_pred = svm.predict(X_train)
    print ('Train accarcy: %f' % np.mean(y_train_pred == Y_train))
    y_val_pred = svm.predict(X_val)
    print ('Val accarcy: %f' % np.mean(y_val_pred == Y_val))


def hyperparameter_tuning():
    """
    """
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()
    learning_rates = [0.75e-7, 1.5e-7, 1.25e-7, 1.75e-7, 2e-7]
    reg_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4, 4.25e4, 4.5e4]
    
    # 结果将以(learning_rate, reg_strength)的形式存在一个字典中
    result = {}
    best_val = -1     # 存储最好的验证集准确度
    best_svm = None   # 存储达到这个最佳值的svm对象
    
    for lr in learning_rates:
        for reg in reg_strengths:
            svm = LinearSVM()
            _ = svm.train(X_train, Y_train, lr, reg, num_iters=1500)
            y_train_pred = svm.predict(X_train)
            accuracy_train = np.mean(Y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            accuracy_val = np.mean(Y_val == y_val_pred)
            result[(lr, reg)] = (accuracy_train, accuracy_val)
            if best_val < accuracy_val:
                best_val = accuracy_val
                best_svm = svm
    
    for lr, reg in sorted(result):
        print ('lr: %e reg: %e train accuracy: %f val accuracy: %f' % 
               (lr, reg, result[(lr, reg)][0], result[(lr, reg)][1]))   
    print ('Best validation accuracy achieved during cross-validation: %f' % best_val)
    
    # 可视化交叉验证结果
    # 可视化训练集准确率
    maker_size = 100
    x_points = [math.log10(x[0]) for x in result]
    y_points = [math.log10(x[1]) for x in result]
    colors = [result[x][0] for x in result]                
    plt.subplot(211)
    plt.scatter(x_points, y_points, maker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate'), plt.ylabel('log reg strength'), plt.title('CIFAR-10 training accuracy')
    
    # 可视化验证集准确率
    colors = [result[x][1] for x in result]
    plt.subplot(212)
    plt.scatter(x_points, y_points, maker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate'), plt.ylabel('log reg strength'), plt.title('CIFAR-10 val accuracy')
    plt.show()
    
    # 在测试集上评价最好的svm的表现
    y_test_pred = best_svm.predict(X_test)
    print ('Test accuracy: %f' % np.mean(y_test_pred == Y_test))
    
    # 可视化best_svm的权重
    W = best_svm.W
    W = np.reshape(W, (32, 32, 3, 10))
    W_min, W_max = np.min(W), np.max(W)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(len(classes)):
        plt.subplot(2, 5, i+1)
        
        # Rescale weights to be bentween 0 and 255
        # squeeze()把W[:, :, :, i]shape中为1的维度去掉
        W_image = 255.0 * (W[:, :, :, i].squeeze() - W_min) / (W_max - W_min)
        plt.imshow(W_image.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()            # plt.show()要放在循环外

if __name__ == '__main__':
    hyperparameter_tuning()
    

         