# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Evaluation metrics
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2020.04.01
Last Update     : 2020.05.05
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

def RMSE(x,y):
    """
    Calculate the RMSE. x is input series and y is reference series.
    It calculates RMSE over the domain, not over time. The spatial structure
    will not be kept.
    ----------------------
    param x: input time series with the shape [time, lat, lon]
    param y: reference time series with the shape [time, lat, lon]
    """
    # error score for temporal-spatial fields, without keeping spatial pattern
    x_series = x.reshape(x.shape[0],-1)
    y_series = y.reshape(y.shape[0],-1)
    rmse = np.sqrt(np.mean((x_series - y_series)**2,1))
    rmse_std = np.sqrt(np.std((x_series - y_series)**2,1))
    
    return rmse, rmse_std

def MAE(x,y):
    """
    Calculate the MAE. x is input series and y is reference series.
    It calculate MAE over time and keeps the spatial structure.
    """
    # error score for temporal-spatial fields, keeping spatial pattern
    mae = np.mean(np.abs(x-y),0)
      
    return mae

def accuracy(pred, label):
    """
    Calculate accuracy score.
    """
    #print("Input size must be [seq, lat, lon]")
    seq, lat, lon = pred.shape
    boolean = (pred==label)
    accu_seq = np.mean(np.mean(boolean.astype(float),2),1)
    accu_spa = np.mean(boolean.astype(float),0)
        
    return accu_seq, accu_spa

# positive is sea ice = 1

def recall(pred, label):
    """
    True positive / Total actual positive
    Input fields must contain only 0 / 1. 1 is positive.
    """
    #print("Input size must be [seq, lat, lon]")
    seq, lat, lon = pred.shape
    # initialize dummy matrix
    pred_dummy_1 = np.zeros(pred.shape,dtype=int)
    label_dummy_1 = np.zeros(label.shape,dtype=int)
    # True positive
    # create dummy matrix to save the labels
    pred_dummy_1[:] = pred[:]
    label_dummy_1[:] = label[:]
    # change the label of negative events
    pred_dummy_1[pred == 0] = 2
    label_dummy_1[label == 0] = 3
    # count True Positive events
    truePositive = (pred_dummy_1 == label_dummy_1)
    
    # initialize dummy matrix
    pred_dummy_2 = np.zeros(pred.shape,dtype=int)
    label_dummy_2 = np.zeros(label.shape,dtype=int)
    # False negative (is 1 but predict 0)
    # create dummy matrix to save the labels (reset dummy)
    pred_dummy_2[:] = pred[:]
    label_dummy_2[:] = label[:]
    pred_dummy_2[pred == 0] = 2
    label_dummy_2[label == 1] = 2
    # count False Positive events
    falseNegative = (pred_dummy_2 == label_dummy_2)
    
#    recall_seq = np.mean(np.mean(np.nan_to_num(truePositive.astype(float) / 
#                                     (truePositive.astype(float) + falseNegative.astype(float))),2),1)
        
    recall_seq = np.sum(np.sum(truePositive.astype(float),2),1) / (np.sum(np.sum(truePositive.astype(float),2),1) +
                                                                    np.sum(np.sum(falseNegative.astype(float),2),1))
        
#    recall_spa = np.mean(np.nan_to_num(truePositive.astype(float) / 
#                                           (truePositive.astype(float) + falseNegative.astype(float))),0)
        
    recall_spa = np.sum(truePositive.astype(float),0) / (np.sum(truePositive.astype(float),0) +
                                                          np.sum(falseNegative.astype(float),0))
        
    #return recall_seq, recall_spa
    return np.nan_to_num(recall_seq), np.nan_to_num(recall_spa)
    
def precision(pred, label):
    """
    True positive / Total predicted positive
    Input fields must contain only 0 / 1. 1 is positive.
    """
    #print("Input size must be [seq, lat, lon]")
    seq, lat, lon = pred.shape
    # initialize dummy matrix
    pred_dummy_1 = np.zeros(pred.shape,dtype=int)
    label_dummy_1 = np.zeros(label.shape,dtype=int)
    # True positive
    # create dummy matrix to save the labels
    pred_dummy_1[:] = pred[:]
    label_dummy_1[:] = label[:]
    # change the label of negative events
    pred_dummy_1[pred == 0] = 2
    label_dummy_1[label == 0] = 3
    # count True Positive events
    truePositive = (pred_dummy_1 == label_dummy_1)

    # initialize dummy matrix
    pred_dummy_2 = np.zeros(pred.shape,dtype=int)
    label_dummy_2 = np.zeros(label.shape,dtype=int)
    # False positive (is 0 but predict 1)
    # create dummy matrix to save the labels (reset dummy)
    pred_dummy_2[:] = pred[:]
    label_dummy_2[:] = label[:]
    pred_dummy_2[pred == 1] = 2
    label_dummy_2[label == 0] = 2
    # count False Positive events
    falsePositive = (pred_dummy_2 == label_dummy_2)
    prec_seq = np.sum(np.sum(truePositive.astype(float),2),1) / (np.sum(np.sum(truePositive.astype(float),2),1) +
                                                                 np.sum(np.sum(falsePositive.astype(float),2),1))
        
    prec_spa = np.sum(truePositive.astype(float),0) / (np.sum(truePositive.astype(float),0) +
                                                       np.sum(falsePositive.astype(float),0))
       
    return np.nan_to_num(prec_seq), np.nan_to_num(prec_spa)

def CRPS(obs, pred, data_structure="sequencial"):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.
    
    The CRPS compares the empirical distribution of an ensemble forecast
    to a scalar observation. Smaller scores indicate better skill.
    
    param obs: deterministic observation with the shape [timesteps]
    param pred: an ensemble of forecast with the shape [ensemble, timesteps]
    param data_structure: structure of data, must be "sequencial" or "spacial"
    
    CRPS is defined for one-dimensional random variables with a probability
    density $p(x)$,
    
    .. math::
        CRPS(F, x) = \int_z (F(z) - H(z - x))^2 dz
        
    where $F(x) = \int_{z \leq x} p(z) dz$ is the cumulative distribution
    function (CDF) of the forecast distribution $F$ and $H(x)$ denotes the
    Heaviside step function, where $x$ is a point estimate of the true
    observation (observational error is neglected).
    
    This function calculates CRPS efficiently using the empirical CDF:
    http://en.wikipedia.org/wiki/Empirical_distribution_function
    """
    if data_structure == "sequencial":
        #print("Input timeseries")
        ens, t = pred.shape
        # sort the forecast matrix following the ensemble axis
        pred_sort = np.sort(pred, axis=0)
        # calculate the Heaviside function
        H = np.zeros(pred.shape,dtype=float)
        # calculate Heaviside function
        obs_h = np.repeat(obs[np.newaxis,:], ens, 0)
        H[pred_sort>obs_h] = 1.0
        # compute the CDF
        cdf_unit = np.arange(1,ens+1,1)
        cdf = np.repeat(cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens
        # calculate dz
        dz = np.zeros(pred.shape,dtype=float)
        dz[1:,:] = pred_sort[1:,:] - pred_sort[:-1,:]
        # calculate CRPS
        CRPS = (cdf - H) ** 2 * dz
        CRPS_int = np.sum(CRPS, 0)
        CRPS_mean = np.mean(CRPS_int)
    elif data_structure == "spatial":
        #print("Input temporal-spatial sequence")
        ens, t, y, x = pred.shape
        # sort the forecast matrix following the ensemble axis
        pred_sort = np.sort(pred, axis=0)
        # calculate the Heaviside function
        H = np.zeros(pred.shape,dtype=float)
        # calculate Heaviside function
        obs_h = np.repeat(obs[np.newaxis,:,:,:], ens, 0)
        H[pred_sort>obs_h] = 1.0
        # compute the CDF
        cdf_unit = np.arange(1,ens+1,1)
        cdf_2D = np.repeat(cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens
        cdf_3D = np.repeat(cdf_2D[:,:,np.newaxis], y, 2)
        cdf = np.repeat(cdf_3D[:,:,:,np.newaxis], x, 3)
        # calculate dz
        dz = np.zeros(pred.shape,dtype=float)
        dz[1:,:,:,:] = pred_sort[1:,:,:,:] - pred_sort[:-1,:,:,:]
        # calculate CRPS
        CRPS = (cdf - H) ** 2 * dz
        CRPS_int = np.sum(CRPS, 0)
        CRPS_mean = np.mean(CRPS_int)
    else:
        raise IOError("The chosen data structure is not supported!")
    
    return CRPS_int, CRPS_mean

def CRPSprob(obs, pred, data_structure="sequencial"):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.
    
    The CRPSprob compares the empirical distribution of an ensemble forecast
    to an "ensemble" of "observation". Such request is always posted by an
    assessment of an ensemble forecast against another ensemble of simulation,
    which is in general known as pseudo observation. Smaller scores indicate
    better skill.
    
    param obs: an ensemble observation with the shape  [ensemble, timesteps]
    param pred: an ensemble of forecast with the shape [ensemble, timesteps]
    param data_structure: structure of data, must be "sequencial" or "spacial"
    
    CRPS is defined for one-dimensional random variables with a probability
    density $p(x)$,
    
    .. math::
        CRPS(F, x) = \int_z (F_z(z) - F_x(z))^2 dz
        
    where $F(x) = \int_{z \leq x} p(z) dz$ is the cumulative distribution
    function (CDF) of the forecast distribution $F_z$ and the pseudo observation.
    
    This function calculates CRPS efficiently using the empirical CDF:
    http://en.wikipedia.org/wiki/Empirical_distribution_function
    """
    if data_structure == "sequencial":
        #print("Input timeseries")
        ens_pred, t = pred.shape
        ens_obs, t = obs.shape
        # sort the forecast matrix following the ensemble axis
        pred_sort = np.sort(pred, axis=0)
        obs_sort = np.sort(obs, axis=0)
        # compute the unit CDF for both
        pred_cdf_unit = np.arange(1,ens_pred+1,1)
        obs_cdf_unit = np.arange(1,ens_obs+1,1)
        pred_cdf = np.repeat(pred_cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens_pred
        obs_cdf = np.repeat(obs_cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens_obs
        # computation of observation CDF multiplied by step length
        dx = np.zeros(obs.shape,dtype=float)
        dx[1:,:] = obs_sort[1:,:] - obs_sort[:-1,:]
        obs_cdf_dx = obs_cdf * dx
        # using a fake dynamic "Heaviside function" to compute the difference between 2 cdf
        # and take the integral
        CRPS_int = np.zeros(t,dtype=float)
        for i in range(ens_pred-1):
            H = np.zeros(obs.shape,dtype=int)
            # initialize H of last timestep
            if i == 0:
                H_laststep = np.zeros(H.shape,dtype=int)
            # take the threshold value for integration, first step is skipped due to dz
            # we take the integral based on slice of forecast
            pred_cri = pred_sort[i+1,:]
            # calculate Heaviside function
            # keep all the values from observation that below the threshold
            pred_h = np.repeat(pred_cri[np.newaxis,:], ens_obs, 0)
            H[pred_h>obs_sort] = 1.0
            # calculate dz
            pred_dz = pred_sort[i+1,:] - pred_sort[i,:]
            # calculate CRPS
            pred_cdf_dz = pred_cdf[i+1,:] * pred_dz
            obs_cdf_sum = np.sum(obs_cdf_dx[:] * H - obs_cdf_dx[:] * H_laststep, 0)
            CRPS = (pred_cdf_dz - obs_cdf_sum) ** 2 / pred_dz # normalized by pred_dz as it is included in the bracket
            # update H for the last time step
            H_laststep[:] = H[:]
            # take the sum of CRPS
            CRPS_int += CRPS
        # take the mean of CRPS
        CRPS_mean = np.mean(CRPS_int)
        
    elif data_structure == "spatial":
        #print("Input temporal-spatial sequence")
        ens_pred, t, y, x = pred.shape
        ens_obs, t, y, x = obs.shape
        # sort the forecast matrix following the ensemble axis
        pred_sort = np.sort(pred, axis=0)
        obs_sort = np.sort(obs, axis=0)
        # compute the unit CDF for both
        pred_cdf_unit = np.arange(1,ens_pred+1,1)
        obs_cdf_unit = np.arange(1,ens_obs+1,1)
        
        pred_cdf_2D = np.repeat(pred_cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens_pred
        pred_cdf_3D = np.repeat(pred_cdf_2D[:,:,np.newaxis], y, 2)
        pred_cdf = np.repeat(pred_cdf_3D[:,:,:,np.newaxis], x, 3)
        
        obs_cdf_2D = np.repeat(obs_cdf_unit[:,np.newaxis], t, 1) * 1.0 / ens_obs
        obs_cdf_3D = np.repeat(obs_cdf_2D[:,:,np.newaxis], y, 2)
        obs_cdf = np.repeat(obs_cdf_3D[:,:,:,np.newaxis], x, 3)
        # computation of observation CDF multiplied by step length
        dx = np.zeros(obs.shape,dtype=float)
        dx[1:,:,:,:] = obs_sort[1:,:,:,:] - obs_sort[:-1,:,:,:]
        obs_cdf_dx = obs_cdf * dx
        # using a fake dynamic "Heaviside function" to compute the difference between 2 cdf
        # and take the integral
        CRPS_int = np.zeros((t, y, x),dtype=float)
        for i in range(ens_pred-1):
            H = np.zeros(obs.shape,dtype=int)
            # initialize H of last timestep
            if i == 0:
                H_laststep = np.zeros(H.shape,dtype=int)  
            # take the threshold value for integration, first step is skipped due to dz
            # we take the integral based on slice of forecast
            pred_cri = pred_sort[i+1,:,:,:]
            # calculate Heaviside function
            # keep all the values from observation that below the threshold
            pred_h = np.repeat(pred_cri[np.newaxis,:,:,:], ens_obs, 0)
            H[pred_h>obs_sort] = 1.0            
            # calculate dz
            pred_dz = pred_sort[i+1,:,:,:] - pred_sort[i,:,:,:]
            # calculate CRPS
            pred_cdf_dz = pred_cdf[i+1,:,:,:] * pred_dz
            obs_cdf_sum = np.sum(obs_cdf_dx[:] * H - obs_cdf_dx[:] * H_laststep, 0)
            CRPS = (pred_cdf_dz - obs_cdf_sum) ** 2 / pred_dz # normalized by pred_dz as it is included in the bracket
            # update H for the last time step
            H_laststep[:] = H[:]
            # take the sum of CRPS
            CRPS_int += CRPS
        # take the mean of CRPS
        CRPS_mean = np.mean(CRPS_int)
    else:
        raise IOError("The chosen data structure is not supported!")
    
    return CRPS_int, CRPS_mean            