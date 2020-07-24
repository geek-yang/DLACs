# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Functions used by the module
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.07.26
Last Update     : 2020.07.16
Contributor     :
Description     : This scripts provides the basic functions be used by other modules, including early-stop,
                  Evidence Lower Bound (ELBO), error weight and density function of Gaussian distribution.
              
                  The early stop module is designed with reference to:
                  https://github.com/Bjarten/early-stopping-pytorch
                  https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py
Return Values   : time series / array
Caveat!         :
"""

import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStop:
    """
    Early stop module is widely used to avoid over-fitting. The loss is assigned to be
    the smaller the better.
    """ 
    def __init__(self, patience: int, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        param patience: how long to wait after last time validation loss improved.
        param verbose: if True, prints a message for each validation loss improvement. 
        param delta: minimum change in the monitored quantity to qualify as an improvement.
        param path: output path of the checkpoint file.
        """         
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
    def __init__(self, train_size, loss_function=nn.MSELoss()):
        """
        Quantify the Evidence Lower Bound (ELBO) and provide the total loss.
        """
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.loss_function = loss_function
        
    def forward(self, input, target, kl, kl_weight=1.0):
        """
        Kullback-Leibler divergence. This comes from
        the euqation (4) in Shridhar et. al. 2019, which consists of likelihood cost
        (is dependent on data) and complexity cost (id dependent on distribution).
        """
        assert not target.requires_grad
        likelihood_cost = self.loss_function(input, target)
        complexity_cost = kl_weight * kl
        total_loss = likelihood_cost + complexity_cost
        return total_loss, likelihood_cost, complexity_cost
