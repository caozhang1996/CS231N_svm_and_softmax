#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:30:18 2018

@author: caozhang
"""

import platform
import numpy as np

version = platform.python_version_tuple()
print version[0]

x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16],
              [17, 18, 19, 20]])

y = np.array([0, 2, 3, 4, 1])
print (y.shape[0])
print (np.zeros_like(y))

num_train = 49000
batch_size = 200
sample_index = np.random.choice(num_train, batch_size, replace=True)
print (sample_index.shape)

