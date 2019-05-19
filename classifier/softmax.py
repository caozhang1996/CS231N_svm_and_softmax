#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 10:11:46 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


 
# 使用循环嵌套实现朴素版softmax损失函数
def softmax_loss_navie(W, B, X, Y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - B: A numpy array of shape (C,) containing biases
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

    Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in xrange(num_train):
        f_i = X[i].dot(W) + B
        # 减去向量中的最大值,避免数值不稳定的问题
        f_i -= np.max(f_i)
        sum_j = np.sum(np.exp(f_i))
        probs = lambda k: np.exp(f_i[k]) / sum_j
        loss += -np.log(probs(Y[i]))
    
        #计算梯度
        for k in xrange(num_classes):
            prob_k = probs(k)
            dW[:, k] += (prob_k - (k == Y[i])) * X[i]   # 每一张图片的梯度都加起来
            
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, B, X, Y, reg):
    num_train = X.shape[0]
    loss = 0.0
    
    f = X.dot(W) + B
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    probs = np.exp(f) / sum_f
    loss = -np.sum(np.log(probs[np.arange(num_train), Y])) / num_train
    
    dW = np.copy(probs)
    dW[np.arange(num_train), Y] -= 1
    dW = (X.T).dot(dW) /num_train
    loss += 0.5 * reg *np.sum(W * W)
    dW += reg * W
    return loss, dW
   
    
        
    
