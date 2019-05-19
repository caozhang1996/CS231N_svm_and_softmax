#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:26:37 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import platform
import pickle


def load_pickle(f):
    """
    """
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('Invalid python version: {}' .format(version))


def load_CIFAR10_batch (file_name):
    """
    """
    with open(file_name, 'r') as f:
        data_dict = load_pickle(f)
        X = data_dict['data']         # 得到10000张图片的平铺
        Y = data_dict['labels']
        X = np.reshape(X, (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y
    

def load_CIFAR10(root):
    """
    """
    X_tr = []
    Y_tr = []
    for i in xrange(1, 6):
        file_name = os.path.join(root, 'data_batch_%d' % i)
        X, Y = load_CIFAR10_batch(file_name)
        X_tr.append(X)
        Y_tr.append(Y)
        
    X_tr = np.concatenate(X_tr)
    Y_tr = np.concatenate(Y_tr)
    X_te, Y_te = load_CIFAR10_batch(os.path.join(root, 'test_batch'))
    return X_tr, Y_tr, X_te, Y_te


    