# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:04:20 2019

@author: dell
"""

from sklearn import preprocessing
import numpy as np
X=np.array([[ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.],
            [ 2.,  1., 3.]])
col_mean=X.mean(axis=0)
col_std=X.std(axis=0)
X_scaled=preprocessing.scale(X)