#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:50:38 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import numpy as np
import matplotlib.pyplot as plt

from  classifier.softmax import *
from classifier.linear_classifier import *
import data_utils


def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=1000, num_dev=500):
    """
    """
    root_dir = '../dataset/cifar-10-batches-py'
    X_train , Y_train, X_test, Y_test = data_utils.load_CIFAR10(root_dir)
    
    mask = list(range(num_training, num_training + num_val))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]
    
    # reshape the images
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # 使X_train, X_dev, X_val, 中的数据零均值化
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_dev -= mean_image
    X_test -= mean_image
    return X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test


def compare_difference():
    """
    """
    W = np.random.randn(3072, 10) * 1e-4    # 1e-4 不能掉,不然W的值很大,将导致一些激活函数饱和
    B = np.zeros(10)
    X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data()
    
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_navie(W, B, X_dev, Y_dev, reg=0.000005)
    toc = time.time()
    print ('naive loss: %e, compute in: %f' % (loss_naive, toc - tic))
    
    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, B, X_dev, Y_dev, reg=0.000005)
    toc = time.time()
    print ('vectorized loss: %e, compute in: %f' % (loss_vectorized, toc - tic))
    
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized)
    print ('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print ('Gradient difference: %f' % grad_difference)


def softmax_train():
    """
    """
    X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data()
    softmax = Softmax()
    tic = time.time()
    loss_history = softmax.train(X_train, Y_train, learning_rate=1e-7, 
                                 reg=2.5e4, num_iters=2000, verbose=True)
    toc = time.time()
    print ('compute in %fs' % (toc - tic))
    
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
    
    # 检测训练得到的W在训练集和验证集上的准确率
    y_train_pred = softmax.predict(X_train)
    print ('Training accuracy: %f' % np.mean(y_train_pred==Y_train))
    y_val_pred = softmax.predict(X_val)
    print ('Validation accuracy: %f' % np.mean(y_val_pred==Y_val))


def hyperparameter_tuning():
    """
    """
    X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data()
    learning_rates = [0.75e-7, 1.5e-7, 1.25e-7, 1.75e-7, 2e-7]
    reg_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4, 4.25e4, 4.5e4]
    
    # 结果将以(learning rate, reg strength)作为k存放在字典中
    results = {}
    best_val = -1
    best_softmax = None
    
    for lr in learning_rates:
        for reg in reg_strengths:
            softmax = Softmax()
            _ = softmax.train(X_train, Y_train, learning_rate=lr, 
                              reg=reg, num_iters=2000, verbose=False)
            y_train_predict = softmax.predict(X_train)
            accuracy_train = np.mean(y_train_predict == Y_train)
            y_val_predict = softmax.predict(X_val)
            accuracy_val = np.mean(y_val_predict == Y_val)
            results[(lr, reg)] = (accuracy_train, accuracy_val)
            if best_val < accuracy_val:
                best_val = accuracy_val
                best_softmax = softmax
    
    for lr, reg in sorted(results):
        print ('lr %e reg %e train accuracy: %f val accuracy: %f'
               % (lr, reg, results[(lr, reg)][0], results[(lr, reg)][1]))
    print ('The best val accuracy is %f' % best_val)
    
    # 在测试集上评价最好的softmax的表现
    y_test_predict = best_softmax.predict(X_test)
    print ('The best test accuracy is %f' % np.mean(y_test_predict == Y_test))
    
    # 可视化交叉验证结果
    # 可视化训练集准确率
    x_points = [math.log10(x[0]) for x in results]  # x轴代表的是learning rate,需要进行取对数操作,因为lr数值太小
    y_points = [math.log10(x[1]) for x in results]  # y轴代表的是reg strength
    maker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(211)
    plt.scatter(x_points, y_points, maker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log10 learning rate '), plt.ylabel('log10 reg strength')
    plt.title('CIFAR-10 train accuracy')
    
    # 可视化验证集准确率
    colors = [results[x][1] for x in results]
    plt.subplot(212)
    plt.scatter(x_points, y_points, maker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log10 learning rate '), plt.ylabel('log10 reg strength')
    plt.title('CIFAR-10 val accuracy')
    plt.show()
    
    # 可视化权重
    W = best_softmax.W
    W = np.reshape(W, (32, 32, 3, 10))
    W_min, W_max = np.min(W), np.max(W)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(len(classes)):
        plt.subplot(2, 5, i+1)
        
        # Rescale weights to be bentween 0 and 255
        W_image = 255.0 * (W[:,:,:,i].squeeze() - W_min) / (W_max - W_min)
        plt.imshow(W_image.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
    

if __name__ == '__main__':
    hyperparameter_tuning()

    
