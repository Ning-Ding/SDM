# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:35:51 2016

@author: dingning
"""

from sklearn.linear_model import Lasso
import numpy as np
from hog_for_coordinates import hog


def train(data, initial_x, N = 5, alpha = 0.1):
    '''
    the standard SDM training function
    ---------------------------------------------------------------------------
    INPUT:
        data: a list containing all the images and true shape coordinates
        initial_x: a numpy array containing initial estimated shape coordinates
        N: number of train iteration
        alpha: the coefficient of L1 regularization of Lasso Linear Regression
    OUTPUT:
        regressors: a list containing a sequence of (R,b)
    ---------------------------------------------------------------------------
    '''
    x = initial_x
    true_x = data[:,1]
    regressors = []
    
    for i in range(N):
        
        '''计算当前坐标与真实坐标之差 '''
        delta_x = true_x - x
        
        
        '''根据当前坐标计算特征矩阵'''
        H = []
        for j in range(len(data)):
            H.append(hog(data[j][0],data[j][1]))
        H = np.array(H)
        
        
        '''
        根据线性最小二乘或回归分析求解(R,b)
        '''
        reg = Lasso(alpha=alpha)
        reg.fit(H,delta_x.ravel())  
        regressors.append([reg.coef_,reg.intercept_])        
                
        '''计算更新之后的坐标x'''
        x = x + reg.coef_ * H + reg.intercept_
        
    return regressors
