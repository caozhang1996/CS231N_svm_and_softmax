#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:34:52 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
    

def svm_loss_navie(W, B, X, Y, reg):
    """
      Structured SVM loss function, naive implementation (with loops).
      
      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
          - W: A numpy array of shape (D, C) containing weights.
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
        scores = X[i].dot(W) + B
        correct_class_score = scores[Y[i]]
        for j in xrange(num_classes):
            if j == Y[i]:
                continue
            else:
                margin = scores[j] - correct_class_score + 1   # delta=1 
                if margin > 0 :
                    loss += margin
                    dW[:, Y[i]] += -X[i, :].T
                    dW[:, j] += X[i, :].T
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    return loss, dW
    

def svm_loss_vectorized(W, B, X, Y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W) + B
    correct_class_scores = scores[np.arange(num_train), Y]
    correct_class_scores = np.reshape(correct_class_scores, (num_train, -1))
    margins = scores - correct_class_scores + 1
    margins = np.maximum(0, margins)
    margins[np.arange(num_train), Y] = 0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    margins[margins > 0] = 1
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), Y] = -row_sum
    dW += np.dot(X.T, margins) / num_train + reg * W
    return loss, dW


    
    
    