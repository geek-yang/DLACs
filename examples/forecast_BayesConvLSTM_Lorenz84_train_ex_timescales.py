# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function    : Forecast Lorenz 84 model - generate forecast with trained BayesConvLSTM model
Author     : Yang Liu
First Built  : 2020.05.09
Last Update  : 2020.05.10
Library     : Pytorth, Numpy, NetCDF4, os, iris, cartopy, dlacs, matplotlib
Description  : This notebook serves to predict the Lorenz 84 model using deep learning. The Bayesian Convolutional
               Long Short Time Memory neural network is used to deal with this spatial-temporal sequence problem.
               We use Pytorch as the deep learning framework.

Return Values  : pkl model and figures

- Lorenz 84 model initial set-up
x=1.0, y=1.0, z=1.0, a=0.25, b=4.0, F=8.0, G=1.0, epsilon=0.4

Study the timescales of Lorenz 84 model and its influence on training and forecast
According to the theory of similarity, timesteps for seasonal and decadal timescales can be calculated based on the unit time scale of Lorenz 84 model, which is 5 day in this case. The relation is shown below:
1 timestep ~4 hours (1/30 5 day)

- 1 time unit = 30 timesteps ~ 5 days 
-------------------------------------------------------------------
- 18 time unit = 540 timesteps ~ seasonal time scale (90 days)
-------------------------------------------------------------------
- 73 time unit = 2160 timesteps ~ annual time scale (365 days)
-------------------------------------------------------------------
- 730 time unit = 21600 timesteps ~ decadal time scale (3650 days)

Note that we count for time units rather than time steps since the system is too simple to compare with actual climate system, therefore we just need to keep the same ratio of unit timestep rather than to have "full cycles" according to the number of timesteps. In a nutshell, there is no real time in the system, only a dimensionless ratio.

In order to study the seasonal and decadal timescales, we take 2920 steps in total and take the last 730 steps as testing data.
"""

import sys
import warnings
import numbers
import logging
import time as tttt

# for data loading
import os
from netCDF4 import Dataset
# for pre-processing and machine learning
import numpy as np
import csv
import sklearn
#import scipy
import torch
import torch.nn.functional

#sys.path.append(os.path.join('C:','Users','nosta','ML4Climate','Scripts','DLACs'))
#sys.path.append("C:\\Users\\nosta\\ML4Climate\\Scripts\\DLACs")
sys.path.append("../")
import dlacs
import dlacs.BayesConvLSTM
import dlacs.preprocess
import dlacs.function
import dlacs.saveNetCDF

# for visualization
import dlacs.visual

# ignore all the DeprecationWarnings by pytorch
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# constants
constant = {'g' : 9.80616,      # gravititional acceleration [m / s2]
            'R' : 6371009,      # radius of the earth [m]
            'cp': 1004.64,      # heat capacity of air [J/(Kg*K)]
            'Lv': 2500000,      # Latent heat of vaporization [J/Kg]
            'R_dry' : 286.9,    # gas constant of dry air [J/(kg*K)]
            'R_vap' : 461.5,    # gas constant for water vapour [J/(kg*K)]
            'rho' : 1026,       # sea water density [kg/m3]
            }

# calculate the time for the code execution
start_time = tttt.time()

################################################################################# 
#########                           datapath                             ########
#################################################################################
# name the type of timescales
timescales = 'unit' # must be "unit" "seasonal" "decadal"
# ** Reanalysis **
# **ERA-Interim** 1979 - 2016 (ECMWF)
# **ORAS4**       1958 - 2014 (ECMWF)
# please specify data path
output_path = '/home/lwc16308/BayesArctic/DLACs/forecast/'
model_path = '/home/lwc16308/BayesArctic/DLACs/models/'
################################################################################# 
#########                             main                               ########
#################################################################################
# set up logging files
logging.basicConfig(filename = os.path.join(output_path,'logFile_forecast_Lorenz84_train_ex_{}.log'.format(timescales)),
                    filemode = 'w+', level = logging.DEBUG,
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

if __name__=="__main__":
    print ('*********************** get the key to the datasets *************************')
    #################################################################################
    ###########                configure Lorenz 84 model                  ###########
    #################################################################################
    logging.info("Configure Lorenz 84 model")

    # Lorenz paramters and initial conditions
    x_init = 1.0 # strength of the symmetric globally encircling westerly current
    y_init = 1.0 # strength of the cosine phases of a chain of superposedwaves (large scale eddies)
    z_init = 1.0 # strength of the sine phases of a chain of superposedwaves (large scale eddies)
    F = 8.0 # thermal forcing term
    G = 1.0 # thermal forcing term
    epsilon = 0.4 # intensity of periodic forcing
    a = 0.25 # stiffness factor for westerly wind x
    b = 4.0 # advection strength of the waves by the westerly current
    
    # assuming the damping time for the waves is 5 days (Lorens 1984)
    dt = 0.0333 # 1/30 unit of time unit (5 days)
    num_steps = 2920
    # cut-off point of initialization period
    cut_off = 0
    
    logging.info("#####################################")
    logging.info("Summary of Lorenz 84 model")
    logging.info("x = 1.0  y = 1.0  z = 1.0")
    logging.info("F = 8.0 G = 1.0 a = 0.25 b = 4.0")
    logging.info("epsilon = 0.4")
    logging.info("unit time step 0.0333 (~5days)")
    logging.info("series length {} steps".format(num_steps))
    logging.info("cut-off length 0 steps") 
    logging.info("target timescales: {}".format(timescales))
    logging.info("#####################################")
    #################################################################################
    ###########            Lorens 84 model + periodic forcing             ###########
    #################################################################################
    def lorenz84_ex(x, y, z, t, a = 0.25, b = 4.0, F = 8.0, G = 1.0, epsilon = 1.0):
        """
        Solver of Lorens-84 model with periodic external forcing.
        
        param x, y, z: location in a 3D space
        param a, b, F, G: constants and forcing
        
        The model is designed with a reference to the paper:
        Broer, H., Simó, C., & Vitolo, R. (2002). Bifurcations and strange
        attractors in the Lorenz-84 climate model with seasonal forcing. Nonlinearity, 15(4), 1205.
        
        Song, Y., Yu, Y., & Wang, H. (2011, October). The stability and chaos analysis of the
        Lorenz-84 atmosphere model with seasonal forcing. In 2011 Fourth International Workshop
        on Chaos-Fractals Theories and Applications (pp. 37-41). IEEE.
        """
        # each time step is ~ 5days, therefore the returning period are 365 / 5 = 73 times in a year
        T = 73
        omega = 2 * np.pi / T
        dx = - y**2 - z**2 - a * x + a * F * (1 + epsilon * np.cos(omega * t))
        dy = x * y - b * x * z - y + G * (1 + epsilon * np.sin(omega * t))
        dz = b * x * y + x * z - z
        
        return dx, dy, dz
    #################################################################################
    ###########        Launch Lorenz 84 model with periodic forcing       ###########
    #################################################################################
    logging.info("Launch Lorenz 84 model")
    # Need one more for the initial values
    x = np.empty(num_steps)
    y = np.empty(num_steps)
    z = np.empty(num_steps)
    t = 0.0
    
    # save initial values
    x[0] = x_init
    y[0] = y_init
    z[0] = z_init
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps-1):
        dx, dy, dz = lorenz84_ex(x[i], y[i], z[i], t, a, b ,F, G, epsilon)
        x[i + 1] = x[i] + (dx * dt)
        y[i + 1] = y[i] + (dy * dt)
        z[i + 1] = z[i] + (dz * dt)
        t += dt
    #################################################################################
    ###########   Preprocess Lorenz 84 model output - low pass filter   ###########
    #################################################################################
    if timescales == "unit":
        win_size = 1
        xx = x
        yy = y
        zz = z
    else:
        if timescales == "seasonal":
            win_size = 18
        if timescales == "annual":
            win_size = 73
        if timescales == "interannual":
            win_size = 365
        elif timescales == "decadal":
            win_size = 730
    
        stat_x = dlacs.preprocess.operator(x[:])
        stat_y = dlacs.preprocess.operator(y[:])
        stat_z = dlacs.preprocess.operator(z[:])
    
        xx = stat_x.lowpass(window=win_size, obj='original')
        yy = stat_y.lowpass(window=win_size, obj='original')
        zz = stat_z.lowpass(window=win_size, obj='original')    
    #################################################################################
    ###########      Prepare Lorenz 84 model output for learning      ###########
    #################################################################################
    print ('*******************  pre-processing  *********************')
    # time series cut-off
    if cut_off:
        xx = xx[cut_off:]
        yy = yy[cut_off:]
        zz = zz[cut_off:]
    print ('===================  normalize data  =====================')
    x_norm = dlacs.preprocess.operator.normalize(xx)
    y_norm = dlacs.preprocess.operator.normalize(yy)
    z_norm = dlacs.preprocess.operator.normalize(zz)
    print('================  save the normalizing factor  =================')
    x_max = np.amax(xx)
    x_min = np.amin(xx)
    y_max = np.amax(yy)
    y_min = np.amin(yy)
    z_max = np.amax(zz)
    z_min = np.amin(zz)
    logging.info("Data preprocessing complete!")
    #################################################################################
    ###########      create basic dimensions for tensor and network       ###########
    #################################################################################
    print ('*******************  create basic dimensions for tensor and network  *********************')
    # specifications of neural network
    input_channels = 3
    #hidden_channels = [3, 2, 1] # number of channels & hidden layers, the channels of last layer is the channels of output, too
    hidden_channels = [3]
    kernel_size = 1
    # here we input a sequence and predict the next step only
    #step = 1 # how many steps to predict ahead
    #effective_step = [0] # step to output
    batch_size = 1
    #num_layers = 1
    learning_rate = 0.01
    num_epochs = 1500
    # check the sequence shape
    sequence_len = num_steps - win_size + 1 - cut_off
    height = 1
    width = 1
    print ('*******************  testing data  *********************')
    # target testing period
    test_len = 730
    print ('*******************  check the environment  *********************')
    print ("Pytorch version {}".format(torch.__version__))
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print("Is CUDA available? {}".format(use_cuda))
    logging.info("Is CUDA available? {}".format(use_cuda))
    # CUDA settings torch.__version__ must > 0.4
    # !!! This is important for the model!!! The first option is gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    #################################################################################
    ###########              load the BayesConvLSTM model               ###########
    #################################################################################
    print ('*******************  load BayesConvLSTM  *********************')
    # load model parameters
    model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels, kernel_size).to(device)
    #model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels, kernel_size, cell_type="full").to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'bayesconvlstm_lorenz84_ex_{}.pkl'.format(timescales)), map_location=device))
    print('##############################################################')
    print('##################  start forecast loop  #####################')
    print('##############################################################')
    # the model learn from time series and try to predict the next time step based on the previous time series
    print ('*******************************  one step ahead forecast  *********************************')
    # time series before test data
    pred_base_x = x_norm[:-test_len]
    pred_base_y = y_norm[:-test_len]
    pred_base_z = z_norm[:-test_len]
    # predict x steps ahead
    step_lead = 16 # unit week
    # ensemble
    ensemble = 20
    # create a matrix for the prediction
    lead_pred_x = np.zeros((test_len,step_lead),dtype=float) # dim [predict time, lead time，lat, lon]
    lead_pred_y = np.zeros((test_len,step_lead),dtype=float) # dim [predict time, lead time，lat, lon]
    lead_pred_z = np.zeros((test_len,step_lead),dtype=float) # dim [predict time, lead time，lat, lon]
    # start the prediction loop
    for ens in range(ensemble):
        print('ensemble No. {}'.format(ens))
        ncKey = dlacs.saveNetCDF.savenc(output_path, 'pred_lorenz84_{}_ens_{}.nc'.format(timescales, ens))
        lead_pred_xyz = np.zeros((3, test_len, step_lead), dtype=float)
        for step in range(test_len):
            # Clear stored gradient
            model.zero_grad()
            # Don't do this if you want your LSTM to be stateful
            # Otherwise the hidden state should be cleaned up at each time step for prediction (we don't clear hidden state in our forward function)
            # see example from (https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py)
            # model.hidden = model.init_hidden()
            # based on the design of this module, the hidden states and cell states are initialized when the module is called.
            for i in np.arange(1,sequence_len - test_len + step + step_lead, 1): 
                #############################################################################
                ###############        before time of prediction        ###############
                #############################################################################
                if i <= (sequence_len - test_len + step):
                    # create variables
                    x_input = np.stack((x_norm[i-1], y_norm[i-1], z_norm[i-1])) #vstack,hstack,dstack
                    x_var_pred = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width),
                                                         requires_grad=False).to(device)
                    # make prediction
                    last_pred, _, _ = model(x_var_pred, i-1, training=False)
                    # record the real prediction after the time of prediction
                    if i == (sequence_len - test_len + step):
                        lead = 0
                        # GPU data should be transferred to CPU
                        lead_pred_x[step,0] = last_pred[0,0].cpu().data.numpy()
                        lead_pred_y[step,0] = last_pred[0,1].cpu().data.numpy()
                        lead_pred_z[step,0] = last_pred[0,2].cpu().data.numpy()
                #############################################################################
                ###############          after time of prediction         ###############
                #############################################################################    
                else:
                    lead += 1
                    # use the predicted data to make new prediction
                    x_input = np.stack((lead_pred_x[step,i-(sequence_len - test_len + step +1)],
                                        lead_pred_y[step,i-(sequence_len - test_len + step +1)],
                                        lead_pred_z[step,i-(sequence_len - test_len + step +1)])) #vstack,hstack,dstack
                    x_var_pred = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width),
                                                         requires_grad=False).to(device)
                    # make prediction
                    last_pred, _, _ = model(x_var_pred, i-1, training=False)
                    # record the prediction
                    lead_pred_x[step,lead] = last_pred[0,0].cpu().data.numpy()
                    lead_pred_y[step,lead] = last_pred[0,1].cpu().data.numpy()
                    lead_pred_z[step,lead] = last_pred[0,2].cpu().data.numpy()
        lead_pred_xyz[0,:,:] = lead_pred_x[:]
        lead_pred_xyz[1,:,:] = lead_pred_y[:]
        lead_pred_xyz[2,:,:] = lead_pred_z[:]
        ncKey.ncfile_Lorenz(lead_pred_xyz)
        logging.info("Saving ncfile No.{}".format(ens))

    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))
    logging.info("--- %s minutes ---" % ((tttt.time() - start_time)/60))
