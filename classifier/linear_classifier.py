#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:39:11 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from classifier.linear_svm import *
from classifier.softmax import *


class LinearClassifier(object):
    """
    """
    def __init__(self):
        self.W = None
        self.B = None
       
        
    def train(self, X, Y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
         Train this linear classifier using stochastic gradient descent.
         使用随机梯度下降来训练这个分类器
         Inputs:
             - X: A numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
             - y: A numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has label 0 <= c < C for C classes.
             - learning_rate: (float) learning rate for optimization.
             - reg: (float) regularization strength.
             - num_iters: (integer) number of steps to take when optimizing
             - batch_size: (integer) number of training examples to use at each step.
             - verbose: (boolean) If true, print progress during optimization.
        Outputs:
             A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(Y) + 1 # 假设y的值是0, 1, ... k-1,其中K是类别数量
        if self.W == None and self.B == None:
            self.W = np.random.randn(dim, num_classes) * 1e-2  # 简易初始化W
            self.B = np.zeros_like(num_classes)
        
        # 运行随机梯度下降法来优化W
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            Y_batch = None
            
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index,:]   # select the batch sample
            Y_batch = Y[sample_index]     # select the batch label
            
            # 评估损失和梯度
            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)
    
            # 参数更新
            self.W += -learning_rate * grad
            
            if verbose and it % 10 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))
                
        return loss_history
    
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:
            - X: A numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.
        Returns:
            - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
            array of length N, and each element is an integer giving the predicted
            class.
        """
        y_pred = np.zeros(X.shape[0])
        score = X.dot(self.W) + self.B
        y_pred = np.argmax(score, axis=1)
        return y_pred
    
    
    def loss(self, X_batch, Y_batch, reg):
        """
        Compute the loss function and its derivative. Subclasses will override this.
        Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
            data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: (float) regularization strength.

        Returns: A tuple containing:
            - loss as a single float
            - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """
    """
    def loss(self, X_batch, Y_batch, reg):
        return svm_loss_vectorized(self.W, self.B, X_batch, Y_batch, reg)
        


class Softmax(LinearClassifier):
    """
    """
    def loss(self, X_batch, Y_batch, reg):
        return softmax_loss_vectorized(self.W, self.B, X_batch, Y_batch, reg)