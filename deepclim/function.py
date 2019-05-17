# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Statistical Operator for Climate Data
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2018.07.26
Last Update     : 2019.05.08
Contributor     :
Description     : This scripts provides the basic functions, which will be used by other modules.
Return Values   : time series / array
Caveat!         :
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

   
def lossPeak(y_pred,y_train,y_max=0.8,y_min=0.3,weight_ex=2):
    """
    Loss function to place high weight on maximum (upper threshold) and minimum (lower threshold) of the training sequence.
    param y_pred: predicted data
    param y_train: training data
    param y_max
    param y_min
    param weight_ex
    """
    error_above = torch.sqrt((y_train[y_train>=y_max] - y_pred[y_train>=y_max]).pow(2).sum())
    error_below = torch.sqrt((y_train[y_train<=y_min] - y_pred[y_train<=y_min]).pow(2).sum())
    error_within = torch.sqrt((y_train[y_min<y_train<y_max] - y_pred[y_min<y_train<y_max]).pow(2).sum())
    error_peak = ((error_above + error_below) * weight_ex + error_within) / (weight_ex * 2 + 1)
    
    return error_peak
    