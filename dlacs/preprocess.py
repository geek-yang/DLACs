# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Statistical Operator for Climate Data
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2018.07.26
Last Update     : 2020.03.17
Contributor     :
Description     : This module provides several methods to perform statistical
                  analysis on MET and all kinds of fields.
Return Values   : time series / array
Caveat!         :
"""

import numpy as np
import scipy
from scipy import stats
import cartopy
import os
import iris
#import matplotlib
#import seaborn as sns
#import matplotlib.pyplot as plt

class operator:
    def __init__(self, var):
        """
        Statistical operations on climate data.
        param var: imput time series
        param outpath: the path for the output files
        """
        self.var = var

    def anomaly(self, Dim_month=True):
        """
        Remove seasonal cycling for monthly data.
        param Dim_month: there are two modes for removing the seasonal cycling
        -True (default) input time series have month dimension [year,month,...]
        -False input time series have only 1 dimension for time
        param white_var: time series without seasonal cycling
        return: time series
        rtype: numpy.array
        """
        # white refers to the time series without seasonal cycling
        white_var = np.zeros(self.var.shape, dtype=float)
        #switch mode
        if Dim_month == True:
            print ('The input data has the dimension of month.')
            # check the dimension of input
            seansonal_cycle_var = np.mean(self.var, axis=0)
            if self.var.ndim == 2:
                t, m = white_var.shape
                for i in np.arange(t):
                        white_var[i,:] = self.var[i,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(t*m)
            elif self.var.ndim == 3:
                t, m, y = white_var.shape
                for i in np.arange(t):
                        white_var[i,:,:] = self.var[i,:,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(t*m,y)
            elif self.var.ndim == 4:
                t, m, y, x = white_var.shape
                for i in np.arange(t):
                        white_var[i,:,:,:] = self.var[i,:,:,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(t*m,y,x)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension higher than 4!")
        else:
            print ('The input data does not have the dimension of month.')
            if self.var.ndim == 1:
                for i in np.arange(12):
                    seansonal_cycle_var = np.mean(self.var[i::12],axis=0)
                    white_var[i::12] = self.var[i::12] - seansonal_cycle_var
            elif self.var.ndim == 2:
                for i in np.arange(12):
                    seansonal_cycle_var = np.mean(self.var[i::12,:],axis=0)
                    white_var[i::12,:] = self.var[i::12,:] - seansonal_cycle_var
            elif self.var.ndim == 3:
                for i in np.arange(12):
                    seansonal_cycle_var = np.mean(self.var[i::12,:,:],axis=0)
                    white_var[i::12,:,:] = self.var[i::12,:,:] - seansonal_cycle_var
            else:
                raise IOError("This module can not work with any array with a \
                              dimension higher than 3!")
        self._anomaly = white_var
        
        print ("The output anomaly time series only contains one dimension for time!")
        
        return self._anomaly

    def detrend(self, order=2, obj='anomaly'):
        """
        Detrend time series through polynomial fit.
        param series: input time series, either 1D or 2/3D
        param order: order of polynomial for fitting
        param obj: objects for detrending, two options available
        -'anomaly' (default) the time series of anomaly will be detrended
        -'original' the original input time series will be detrended
        return: time series
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            print ("Make sure that the input time series has only 1 dimension for time!")
            series = self.var
        else:
            raise IOError("Please choose the right input mode for detrending!")
        # check the dimension of input
        if series.ndim == 1:
            polynomial = np.polyfit(np.arange(len(series)), series, order)
            poly_fit = np.poly1d(polynomial)
            poly_fit_var = poly_fit(np.arange(len(series)))
        elif series.ndim == 2:
            poly_fit_var = np.zeros(series.shape, dtype=float)
            t, y = poly_fit_var.shape
            for i in np.arange(y):
                polynomial = np.polyfit(np.arange(t), series[:,i], order)
                poly_fit = np.poly1d(polynomial)
                poly_fit_var[:,i] = poly_fit(np.arange(t))
        elif series.ndim == 3:
            poly_fit_var = np.zeros(series.shape, dtype=float)
            t, y, x = poly_fit_var.shape
            for i in np.arange(y):
                for j in np.arange(x):
                    polynomial = np.polyfit(np.arange(t), series[:,i,j], order)
                    poly_fit = np.poly1d(polynomial)
                    poly_fit_var[:,i,j] = poly_fit(np.arange(t))
        else:
            raise IOError("This module can not work with any array with a \
                            dimension higher than 3!")
        self._polyfit = poly_fit_var
        self._detrend = series - self._polyfit

        return self._detrend
    
    def lowpass(self, window=60, obj='anomaly'):
        """
        Apply low pass filter to the time series. The function gives running mean
        for the point AT The End Of The Window!!
        param series: input time series, either 1D or 2/3D
        param window: time span for the running mean
        param obj: object for detrending, two options available
        -'anomaly' (default) apply low pass filter to the time series of anomaly
        -'original' apply lowpass filter to the original input time series
        -'detrend' apply lowpass filter to the detrended time series
        return: time series
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            series = self.var
        elif obj == 'detrend':
            series = self._detrend        
        # check the dimension of input
        if series.ndim == 1:
            t = len(series)
            running_mean = np.zeros(t-window+1, dtype=float)
            for i in np.arange(t-window+1):
                running_mean[i] = np.mean(series[i:i+window])
        elif series.ndim == 2:
            t, y  = series.shape
            running_mean = np.zeros((t-window+1, y), dtype=float)
            for i in np.arange(t-window+1):
                running_mean[i,:] = np.mean(series[i:i+window,:],0)
        elif series.ndim == 3:
            t, y, x = series.shape
            running_mean = np.zeros((t-window+1, y, x), dtype=float)
            for i in np.arange(t-window+1):
                running_mean[i,:,:] = np.mean(series[i:i+window,:,:],0)
        else:
            raise IOError("This module can not work with any array with a \
                            dimension higher than 3!")
        self._lowpass = running_mean
        
        return self._lowpass
    
    @staticmethod
    def autoCorr(series):
        """
        Quantify the autocorrelation og given time series.
        param series: inout time series, should be 1D
        return: A series of correlation coefficient w.r.t lag time.
        rtype: numpy.array
        """
        # normalize the signal
        series_norm = np.sum(series**2)
        # calculate the autocorrelation
        auto_corr = np.correlate(series, series, 'full') / series_norm
        # for the return value, we only keep half of them
        return auto_corr[len(auto_corr)//2:]
    
    @staticmethod
    def linearRegress(var_x, var_y, lag=0):
        """
        Linear regression of input time series. Lead/lag regression can also be performed.
        param var_x: input time series, either 1D or 2D
        param var_y: input time series as the regression target, either 1D or 3D
        param lag: time unit for lead / lag regression, lag must be an integer
        return: matrix of slope, regression coefficient and rate of non-trustworthy
        rtype: numpy.array
        """
        # check the dimensions of input time series
        if var_x.shape == var_y.shape:
            if var_y.ndim == 2:
                print("One time series with 2 dimensions is regressed on another.")
                if lag == 0:
                    t, y  = var_y.shape
                    slope = np.zeros(y, dtype=float)
                    r_value = np.zeros(y, dtype=float)
                    p_value = np.zeros(y, dtype=float)
                    for i in np.arange(y):
                        slope[i], _, r_value[i], p_value[i], _ = stats.linregress(var_x[:,i], var_y[:,i])
                elif lag > 0:
                    print ("This a regression with lead/lag analysis.")
                    t, y  = var_y.shape
                    lag_index = np.arange(-lag,lag+1,1)
                    slope = np.zeros((len(lag_index),y), dtype=float)
                    r_value = np.zeros((len(lag_index),y), dtype=float)
                    p_value = np.zeros((len(lag_index),y), dtype=float)
                    # regress
                    for i in np.arange(len(lag_index)):
                        for j in np.arange(y):
                            if lag_index[i]<0: # var_x lead var_y
                                slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                               var_x[:lag_index[i],j], var_y[-lag_index[i]:,j])
                            elif lag_index[i]>0: # var_y lead var_x
                                slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                               var_x[lag_index[i]:,j], var_y[:-lag_index[i],j])
                            else:
                                slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                               var_x[:,j], var_y[:,j])
                else:
                    IOError("The lead / lag coefficient should be positive integers.")
            elif var_y.ndim == 3:
                print("One time series with 3 dimensions is regressed on another with the same dimensions.")
                if lag == 0:
                    t, y, x  = var_y.shape
                    slope = np.zeros((y, x), dtype=float)
                    r_value = np.zeros((y, x), dtype=float)
                    p_value = np.zeros((y, x), dtype=float)
                    for i in np.arange(y):
                        for j in np.arange(x):
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x[:,i,j], var_y[:,i,j])
                elif type(lag) == int:
                    t, y, x  = var_y.shape
                    slope = np.zeros((y, x), dtype=float)
                    r_value = np.zeros((y, x), dtype=float)
                    p_value = np.zeros((y, x), dtype=float)
                    for i in np.arange(y):
                        for j in np.arange(x):
                            if lag > 0:
                                slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x[lag:,i,j],
                                                                                                var_y[:-lag,i,j])
                            elif lag < 0:
                                slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x[:lag,i,j],
                                                                                                var_y[-lag:,i,j])
                else:
                    IOError("The lead / lag coefficient should be integers.")
            else:
                raise IOError("The dimensions of input time series are not supported!")
        elif var_y.ndim == 2 and var_x.ndim == 1:
            print("One time series is regressed on a 1D field.")
            if lag == 0:
                t, y  = var_y.shape
                slope = np.zeros(y, dtype=float)
                r_value = np.zeros(y, dtype=float)
                p_value = np.zeros(y, dtype=float)
                for i in np.arange(y):
                    slope[i], _, r_value[i], p_value[i], _ = stats.linregress(var_x[:], var_y[:,i])
            elif lag > 0:
                print ("This a regression with lead/lag analysis.")
                t, y  = var_y.shape
                lag_index = np.arange(-lag,lag+1,1)
                slope = np.zeros((len(lag_index),y), dtype=float)
                r_value = np.zeros((len(lag_index),y), dtype=float)
                p_value = np.zeros((len(lag_index),y), dtype=float)
                # regress
                for i in np.arange(len(lag_index)):
                    for j in np.arange(y):
                        if lag_index[i]<0: # var_x lead var_y
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                            var_x[:lag_index[i]], var_y[-lag_index[i]:,j])
                        elif lag_index[i]>0: # var_y lead var_x
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                            var_x[lag_index[i]:], var_y[:-lag_index[i],j])
                        else:
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(
                                            var_x[:], var_y[:,j])                
        elif var_y.ndim == 3 and var_x.ndim == 1:
            print("One time series is regressed on a 2D field.")
            if lag == 0:
                t, y, x  = var_y.shape
                slope = np.zeros((y, x), dtype=float)
                r_value = np.zeros((y, x), dtype=float)
                p_value = np.zeros((y, x), dtype=float)
                for i in np.arange(y):
                    for j in np.arange(x):
                        slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x, var_y[:,i,j])
            elif type(lag) == int:
                print ("This a regression with lead/lag analysis.")
                print ("Positive lag means 2nd input leads 1st, vice versa.")
                t, y, x  = var_y.shape
                slope = np.zeros((y, x), dtype=float)
                r_value = np.zeros((y, x), dtype=float)
                p_value = np.zeros((y, x), dtype=float)
                for i in np.arange(y):
                    for j in np.arange(x):
                        if lag > 0:
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x[lag:],
                                                                                            var_y[:-lag,i,j])
                        elif lag < 0:
                            slope[i,j], _, r_value[i,j], p_value[i,j], _ = stats.linregress(var_x[:lag],
                                                                                            var_y[-lag:,i,j])                           
            else:
                IOError("The lead / lag coefficient should be integers.")
        else:
            IOError("The dimensons of input time series are not supported.")
        
        return slope, r_value, p_value

    @staticmethod
    def seasons(series, span='DJF', Dim_month=False):
        """
        Extract time series for certain months from given series.
        The given time series should include the time series of all seasons, starting
        from January to December.
        The module extracts 3 month per year based on given argument to incoorporate
        with lead / lag regressions with following modules.
        param series: input time series containing the data for all seasons.
        param span: Target months for data extraction. Following options are available:
        - DJF (default) December, January and February (winter)
        - JJA June, July, August (summer)
        - NDJ November, December and January
        - OND October, November, December
        - SON September, October, November (autumn)
        - MJJ May, June, July
        - AMJ April, May, June 
        - MAM March, April, May (spring)
        param Dim_month: A check whether the time series include the dimension of month.
        return: time series
        rtype: numpy.array
        """
        # check if the input time is in the pre-defined month list
        month_list = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                      'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        if span not in month_list:
            raise IOError("The input month span does not include 3 contineous calander months!")
        # rearange the input series
        if Dim_month == True:
            if series.ndim == 2:
                t, m = series.shape
                series = series.reshape(t*m)
            elif series.ndim == 3:
                t, m, y = series.shape
                series = series.reshape(t*m, y)
            elif series.ndim == 4:
                t, m, y, x = series.shape
                series = series.reshape(t*m, y, x)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension higher than 4!")
        else:
            pass
        # select the months for extraction
        month_1 = 0
        # months combinations except 'DJF' 'NDJ'
        if span == 'JJA':
            month_1 = 6
        elif span == 'OND':
            month_1 = 10
        elif span == 'SON':
            month_1 = 9
        elif span == 'ASO':
            month_1 = 8
        elif span == 'JAS':
            month_1 = 7            
        elif span == 'MJJ':
            month_1 = 5
        elif span == 'AMJ':
            month_1 = 4
        elif span == 'MAM':
            month_1 = 3
        elif span == 'FMA':
            month_1 = 2
        elif span == 'JFM':
            month_1 = 1
        month_2 = month_1 + 1
        month_3 = month_1 + 2
        # now we deal with the exception
        if span == 'DJF':
            month_1 = 1
            month_2 = 2
            month_3 = 12
        elif span == 'NDJ':
            month_1 = 1
            month_2 = 11
            month_3 = 12
        # seperate summer and winter from the rest of the months
        if series.ndim == 1:
            t = len(series)
            series_season = np.zeros(t//4,dtype=float)
            series_season[0::3] = series[month_1-1::12]
            series_season[1::3] = series[month_2-1::12]
            series_season[2::3] = series[month_3-1::12]
        elif series.ndim == 2:
            t, y = series.shape
            series_season = np.zeros((t//4,y),dtype=float)
            series_season[0::3,:] = series[month_1-1::12,:]
            series_season[1::3,:] = series[month_2-1::12,:]
            series_season[2::3,:] = series[month_3-1::12,:]        
        elif series.ndim == 3:
            t, y, x = series.shape
            series_season = np.zeros((t//4,y,x),dtype=float)
            series_season[0::3,:,:] = series[month_1-1::12,:,:]
            series_season[1::3,:,:] = series[month_2-1::12,:,:]
            series_season[2::3,:,:] = series[month_3-1::12,:,:]
        else:
            raise IOError("This module can not work with any array with a \
                           dimension higher than 3!")
        return series_season
        
    @staticmethod
    def interpolation(series, lat_nav, lat_tar, interp_kind='slinear',Dim_month=True):
        """
        Interpolate a time series onto certain latitudes for the coupling
        and comparison of atmosphere and ocean fields.
        It is recommended to interpolate time series of oceanic fields on
        the latitude of the certain atmospheric fields to avoid the data
        out of range issues.
        param series: input time series
        param lat_nav: original latitude for the input data
        param lat_tar: target latitude for interpolation
        param interp_kind: the methods for interpolation, it includse
        -linear
        -nearest
        -(spline)slinear (default) / quadratic / cubic
        param Dim_month: there are two modes for removing the seasonal cycling
        -True (default) input time series have month dimension [year,month,...]
        -False input time series have only 1 dimension for time
        return: time series
        rtype: numpy.array
        """
        if series.ndim > 3:
            raise IOError("This module can not work with any array with a \
                           dimension higher than 3!")
        else:
            if Dim_month == True:
                t, m, y = series.shape
                interp_series = np.zeros((t, m, len(lat_tar)), dtype=float)
                for i in np.arange(t):
                    for j in np.arange(m):
                        # call the data attribute in case it has mask
                        ius = scipy.interpolate.interp1d(lat_nav.data, series[i,j,:], kind=interp_kind,
                                                         bounds_error=False, fill_value=0.0)
                        interp_series[i,j,:] = ius(lat_tar.data)
            else:
                t, y = series.shape
                interp_series = np.zeros((t, len(lat_tar)), dtype=float)
                for i in np.arange(t):
                    ius = scipy.interpolate.interp1d(lat_nav.data, series[i,:], kind=interp_kind,
                                                     bounds_error=False, fill_value=0.0)
                    interp_series[i,:] = ius(lat_tar.data)
            
        return interp_series
    
    @staticmethod
    def unfold(series):
        """
        Reshape the input matrix into time series.
        param Dim_month: there are two modes for removing the seasonal cycling
        Input time series should have month dimension [year,month,...]
        return: time series
        rtype: numpy.array
        """
        if series.ndim == 2:
            t, m = series.shape
            series_out = series.reshape(t*m)
        elif series.ndim == 3:
            t, m, y = series.shape
            series_out = series.reshape(t*m,y)
        elif series.ndim == 4:
            t, m, y, x = series.shape
            series_out = series.reshape(t*m,y,x)
        else:
            raise IOError("This module can not work with any array with a \
                          dimension higher than 4!")
        
        return series_out
    
    @staticmethod
    def normalize(series):
        """
        Normalize the input series.
        """
        max_value = np.amax(series)
        min_value = np.amin(series)
        series_out = np.zeros(series.shape,dtype=float)
        for i in np.arange(len(series)):
            series_out[i] = (series[i] - min_value)/(max_value - min_value)
        
        return series_out
    
    @staticmethod
    def denormalize(series_norm, series_ori):
        """
        Denormalize the input series.
        """        
        max_value = np.amax(series_ori)
        min_value = np.amin(series_ori)
        series_out = np.zeros(series_norm.shape,dtype=float)
        for i in np.arange(len(series_norm)):
            series_out[i] = series_norm[i]*(max_value - min_value) + min_value
            
        return series_out
            
    
    @staticmethod
    def mca(matrix_a, matrix_b, dimension = 'spatial'):
        """
        Perform Maximum Covariance Analysis (MCA) with given data.
        The MCA is based on Singular Value Decomposition (SVD).
        return: eigenvalues and eigenvectors from Singular Value Decomposition.
        rtype: numpy.array
        param matrix_a: first array for SVD
        param matrix_b: second array for SVD
        param dimension: decide which dimension (EOF) to keep, 2 options available:
                         "spatial" / "temporal"
        """
        print ("The input matrix should only have 3 dimensions including time as the first axis.")
        if matrix_a.ndim == 3 and matrix_b.ndim == 3:
            nt_a, ny_a, nx_a = matrix_a.shape
            nt_b, ny_b, nx_b = matrix_b.shape
        else:
            print ("The dimensions of input arrays do not satisfy the requirement!")
        # reshape the input arrays
        matrix_a_2D = np.reshape(matrix_a,[nt_a, ny_a*nx_a], order='F')
        matrix_b_2D = np.reshape(matrix_b,[nt_b, ny_b*nx_b], order='F')
        # choose which dimension to keep
        if dimension == "spatial":
            # calculate the covariance matrix
            # time should be the 1st axis!
            covariance_a_b = np.dot(matrix_a_2D.T, matrix_b_2D) / (nt_a-1)
            # apply the SVD
            U, sigma, V = np.linalg.svd(covariance_a_b, full_matrices=False,
                                        compute_uv = True)
            # Take the 1st mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D, U[:,0])
            mca_left_pattern_1 = U[:,0] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D, V[0,:].T)
            mca_right_pattern_1 = V[0,:] * np.std(mca_b_series)
            # Take the 2nd mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D, U[:,1])
            mca_left_pattern_2 = U[:,1] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D, V[1,:].T)
            mca_right_pattern_2 = V[1,:] * np.std(mca_b_series)
            # Take the 2nd mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D, U[:,2])
            mca_left_pattern_3 = U[:,2] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D, V[2,:].T)
            mca_right_pattern_3 = V[2,:] * np.std(mca_b_series)
            
        elif dimension == "temporal":
            covariance_a_b = np.dot(matrix_a_2D, matrix_b_2D.T) / (ny_a*nx_a-1)
            U, sigma, V = np.linalg.svd(covariance_a_b, full_matrices=False,
                                        compute_uv = True)
            # Take the 1st mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D.T, U[:,0])
            mca_left_pattern_1 = U[:,0] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D.T, V[0,:].T)
            mca_right_pattern_1 = V[0,:] * np.std(mca_b_series)
            # Take the 2nd mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D.T, U[:,1])
            mca_left_pattern_2 = U[:,1] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D.T, V[1,:].T)
            mca_right_pattern_2 = V[1,:] * np.std(mca_b_series)
            # Take the 2nd mode of left pattern and right pattern and plot
            mca_a_series = np.dot(matrix_a_2D.T, U[:,2])
            mca_left_pattern_3 = U[:,2] * np.std(mca_a_series)
            
            mca_b_series = np.dot(matrix_b_2D.T, V[2,:].T)
            mca_right_pattern_3 = V[2,:] * np.std(mca_b_series)
        else:
            raise IOError("The dimension input is not correct!")
        # fraction of squared covariance explained
        # It tells how much squared covariance is expalined by each mode
        sc = sigma**2 / np.sum(sigma**2)

        if dimension == "spatial":
            # reshape mca matrix
            mca_left_pattern_1 = np.reshape(mca_left_pattern_1, 
                                            [ny_a, nx_a], order='F')
            mca_right_pattern_1 = np.reshape(mca_right_pattern_1,
                                             [ny_b, nx_b], order='F')
            mca_left_pattern_2 = np.reshape(mca_left_pattern_2, 
                                            [ny_a, nx_a], order='F')
            mca_right_pattern_2 = np.reshape(mca_right_pattern_2,
                                             [ny_b, nx_b], order='F')
            mca_left_pattern_3 = np.reshape(mca_left_pattern_3, 
                                            [ny_a, nx_a], order='F')
            mca_right_pattern_3 = np.reshape(mca_right_pattern_3,
                                             [ny_b, nx_b], order='F')
        
        return sc, mca_left_pattern_1, mca_right_pattern_1, mca_left_pattern_2, mca_right_pattern_2, mca_left_pattern_3, mca_right_pattern_3
    
def week2month(series, m):
    """
    Select certain month from yearly data at weekly resolution.
    Parameters
    ----------
    series : array-like
        Three-dimensional numeric arrays with time as the first dimenison [time, lat, lon]
    m: int
        Month (from 1 to 12).
    """
    time_year, lat, lon = series.shape
    time_month = time_year // 12
    series_month = np.zeros((time_month, lat, lon), dtype=float)
    series_month[::4,:,:] = series[(m-1)*4::48,:,:]
    series_month[1::4,:,:] = series[(m-1)*4+1::48,:,:]
    series_month[2::4,:,:] = series[(m-1)*4+2::48,:,:]
    series_month[3::4,:,:] = series[(m-1)*4+3::48,:,:]
    
    return series_month