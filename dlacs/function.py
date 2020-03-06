# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Functions used by the module
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.07.26
Last Update     : 2020.03.06
Contributor     :
Description     : This scripts provides the basic functions, which will be used by other modules.
Return Values   : time series / array
Caveat!         :
"""

import math
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

def logpdf_Gaussian(weight, mu=0.0, std=0.25):
    """
    Entropy of probability density function of Gaussian distribution. This function will 
    be used to compute the entropy of Gaussian sampling by posterior and prior.
    log(f(x)) = log(1/(sigma sqrt(2pi) * e^(-1/2 * ((x-mu)/sigma)^2))
    For posterior, mean and log_std are trainable variables. While for prior, mean and log_var
    are constants (fixed Gaussian distribution).
    Note that 
    param weight: weight matrix after sampling the Gaussian distribution
    param mu: mean value of Gaussian distribution
    param std: standard deviation of Gaussian distribution
    """
    if isinstance(std, float):
        entropy = - math.log(std) - math.log(math.sqrt(2*math.pi)) - (weight - mu)**2 / (2 * std**2)
    else:
        entropy = - torch.log(std) - math.log(math.sqrt(2*math.pi)) - (weight - mu)**2 / (2 * std**2)

    return entropy

class ELBO(nn.Module):
    def __init__(self, train_size, loss_function=nn.KLDivLoss()):
        """
        Quantify the Evidence Lower Bound (ELBO) and provide the total loss.
        """
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.loss_function = loss_function
        
    def forward(self, input, target, kl, kl_weight=1.0):
        """
        Kullback-Leibler divergence. This comes from
        the euqation (4) in Shridhar et. al. 2019, where the first term is
        indeed the likelihood cost, and the second term is the complexity cost.
        """
        assert not target.requires_grad
        return self.loss_function(input, target) * self.train_size + kl_weight * kl