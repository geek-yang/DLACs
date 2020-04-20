# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function     : Forecast Lorenz 84 model - Train BayesConvLSTM model
Author       : Yang Liu
First Built  : 2020.03.09
Last Update  : 2020.04.12
Library      : Pytorth, Numpy, NetCDF4, os, iris, cartopy, dlacs, matplotlib
Description  : This notebook serves to predict the Lorenz 84 model using deep learning. The Bayesian Convolutional
               Long Short Time Memory neural network is used to deal with this spatial-temporal sequence problem.
               We use Pytorch as the deep learning framework.

Return Values   : pkl model and figures
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

# for visualization
import dlacs.visual
import matplotlib
# Generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import iris # also helps with regriding
import cartopy
import cartopy.crs as ccrs

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
# ** Reanalysis **
# **ERA-Interim** 1979 - 2016 (ECMWF)
# **ORAS4**       1958 - 2014 (ECMWF)
# please specify data path
datapath = '/projects/0/blueactn/dataBayes'
output_path = '/home/lwc16308/BayesArctic/DLACs/models/'
################################################################################# 
#########                             main                               ########
#################################################################################
# set up logging files
logging.basicConfig(filename = os.path.join(output_path,'logFile_Lorenz84_train.log'),
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
    a = 0.25 # stiffness factor for westerly wind x
    b = 4.0 # advection strength of the waves by the westerly current
    
    # assuming the damping time for the waves is 5 days (Lorens 1984)
    dt = 0.0333 # 1/30 unit of time unit (5 days)
    num_steps = 1500
    # cut-off point of initialization period
    cut_off = 300
    
    logging.info("#####################################")
    logging.info("Summary of Lorenz 84 model")
    logging.info("x = 1.0  y = 1.0  z = 1.0")
    logging.info("F = 8.0 G = 1.0 a = 0.25 b = 4.0")
    logging.info("unit time step 0.0333 (~5days)")
    logging.info("series length 1500 steps")
    logging.info("cut-off length 300 steps")    
    logging.info("#####################################")
    #################################################################################
    ###########                     Lorens 84 model                       ###########
    #################################################################################    
    def lorenz84(x, y, z, a = 0.25, b = 4.0, F = 8.0, G = 1.0):
        """
        Solver of Lorens-84 model.
        param x, y, z: location in a 3D space
        param a, b, F, G: constants and forcing
        """
        dx = - y**2 - z**2 - a * x + a * F
        dy = x * y - b * x * z - y + G
        dz = b * x * y + x * z - z
        
        return dx, dy, dz    
    #################################################################################
    ###########                 Launch Lorenz 84 model                    ###########
    #################################################################################
    logging.info("Launch Lorenz 84 model")
    # Need one more for the initial values
    x = np.empty(num_steps)
    y = np.empty(num_steps)
    z = np.empty(num_steps)
    
    # save initial values
    x[0] = x_init
    y[0] = y_init
    z[0] = z_init
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps-1):
        dx, dy, dz = lorenz84(x[i], y[i], z[i])
        x[i + 1] = x[i] + (dx * dt)
        y[i + 1] = y[i] + (dy * dt)
        z[i + 1] = z[i] + (dz * dt)    
    #################################################################################
    ###########        Prepare Lorenz 84 model output for learning        ###########
    #################################################################################
    # time series cut-off
    x = x[cut_off:]
    y = y[cut_off:]
    z = z[cut_off:]
    print ('===================  normalize data  =====================')
    x_norm = dlacs.preprocess.operator.normalize(x)
    y_norm = dlacs.preprocess.operator.normalize(y)
    z_norm = dlacs.preprocess.operator.normalize(z)
    print('================  save the normalizing factor  =================')
    x_max = np.amax(x)
    x_min = np.amin(x)
    y_max = np.amax(y)
    y_min = np.amin(y)
    z_max = np.amax(z)
    z_min = np.amin(z)
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
    num_epochs = 1000
    print ('*******************  testing data  *********************')
    test_len = 200
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
    ###########              create the BayesConvLSTM model               ###########
    #################################################################################
    print ('*******************  run BayesConvLSTM  *********************')
    print ('The model is designed to make many to one prediction.')
    print ('A series of multi-chanel variables will be input to the model.')
    print ('The model learns by verifying the output at each timestep.')
    # check the sequence shape
    sequence_len = num_steps - cut_off
    height = 1
    width = 1
    # initialize our model
    #model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels, kernel_size).to(device)
    model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels,
                                              kernel_size, cell_type="full").to(device)
    # use Evidence Lower Bound (ELBO) to quantify the loss
    ELBO = dlacs.function.ELBO(height*width)
    # penalty for kl
    penalty_kl = 10
    # stochastic gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Adam optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    print(ELBO)
    print(optimiser)
    print('##############################################################')
    print('##################  start training loop  #####################')
    print('##############################################################')
    # track training loss
    hist = np.zeros(num_epochs)
    hist_likelihood = np.zeros(num_epochs)
    hist_complexity = np.zeros(num_epochs)
    # loop of epoch
    for t in range(num_epochs):
        # Clear stored gradient
        model.zero_grad()
        # loop of timestep
        for timestep in range(sequence_len - test_len):
            # hidden state re-initialized inside the model when timestep=0
            #################################################################################
            ########          create input tensor with multi-input dimension         ########
            #################################################################################
            # create variables
            x_input = np.stack((x_norm[timestep], y_norm[timestep], z_norm[timestep])) #vstack,hstack,dstack
            x_var = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width)).to(device)
            #################################################################################
            ########       create training tensor with multi-input dimension         ########
            #################################################################################
            y_train_stack = np.stack((x_norm[timestep+1], y_norm[timestep+1], z_norm[timestep+1])) #vstack,hstack,dstack
            y_var = torch.autograd.Variable(torch.Tensor(y_train_stack).view(-1,hidden_channels[-1],height,width)).to(device)
            #################################################################################   
            # Forward pass
            y_pred, kl_loss, _ = model(x_var, timestep)
            # choose training data
            y_target = y_var
            # torch.nn.functional.mse_loss(y_pred, y_train) can work with (scalar,vector) & (vector,vector)
            # Please Make Sure y_pred & y_train have the same dimension
            # accumulate loss
            if timestep == 0:
                loss, likelihood, complexity = ELBO(y_pred, y_target, kl_loss,
                                                    1 / (len(hidden_channels) * 8 * penalty_kl * kernel_size**2))
            else:
                loss_step, likelihood_step,\
                complexity_step = ELBO(y_pred, y_target, kl_loss,
                                       1 / (len(hidden_channels) * 8 * penalty_kl * kernel_size**2))
                loss += loss_step
                likelihood += likelihood_step
                complexity += complexity_step
        #print(y_pred.shape)
        #print(y_train.shape)
        # print loss at certain iteration
        if t % 10 == 0:
            print("Epoch {} ELBO: {:0.3f}".format(t, loss.item()))
            logging.info("Epoch {} MSE: {:0.3f}".format(t,loss.item()))
            print("likelihood cost {:0.3f} #*# complexity cost {:0.3f}".format(likelihood.item(), complexity.item()))
            logging.info("likelihood cost {:0.3f} #*# complexity cost {:0.3f}".format(likelihood.item(), complexity.item()))
            print("=========================================")
            logging.info("==========================================")
            # gradient check
            # Gradcheck requires double precision numbers to run
            #res = torch.autograd.gradcheck(loss_fn, (y_pred.double(), y_train.double()), eps=1e-6, raise_exception=True)
            #print(res)
        hist[t] = loss.item()
        hist_likelihood[t] = likelihood.item()
        hist_complexity[t] = complexity.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()
    
        # Backward pass
        loss.backward()

        # Update parametersdd
        optimiser.step()
        
    # save the model
    # (recommended) save the model parameters only
    torch.save(model.state_dict(), os.path.join(output_path,'bayesconvlstm_lorenz84.pkl'))
    # save the entire model
    # torch.save(model, os.path.join(output_path,'bayesconvlstm.pkl'))
    #################################################################################
    ###########                 after training statistics                 ###########
    #################################################################################
    print ("*******************  Loss with time  **********************")
    fig00 = plt.figure()
    plt.plot(hist, 'r', label="Training loss")
    plt.plot(hist_likelihood, 'g', label="Likelihood loss")
    plt.plot(hist_complexity, 'b', label="Complexity loss")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    fig00.savefig(os.path.join(output_path,'BayesConvLSTM_pred_error_Lorenz84.png'),dpi=150)
    
    print ("*******************  Loss with time (log)  **********************")
    fig01 = plt.figure()
    plt.plot(np.log(hist), 'r', label="Training loss")
    plt.plot(np.log(hist_likelihood), 'g', label="Likelihood loss")
    plt.plot(np.log(hist_complexity), 'b', label="Complexity loss")
    plt.xlabel('Epoch')
    plt.ylabel('Log error')
    plt.legend()
    plt.show()
    fig01.savefig(os.path.join(output_path,'BayesConvLSTM_pred_log_error_Lorenz84.png'),dpi=150)
    
    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))
    logging.info("--- %s minutes ---" % ((tttt.time() - start_time)/60))
