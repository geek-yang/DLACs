# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function     : Predict the Spatial Sea Ice Concentration with BayesConvLSTM at weekly time scale - Train model
Author       : Yang Liu
First Built  : 2020.03.09
Last Update  : 2020.03.12
Library      : Pytorth, Numpy, NetCDF4, os, iris, cartopy, dlacs, matplotlib
Description  : This notebook serves to predict the Arctic sea ice using deep learning. The Bayesian Convolutional
               Long Short Time Memory neural network is used to deal with this spatial-temporal sequence problem.
               We use Pytorch as the deep learning framework.

Here we predict sea ice concentration with one extra relevant field from either ocean or atmosphere to test the predictor.

Return Values   : pkl model and figures

The regionalization adopted here follows that of the MASIE (Multisensor Analyzed Sea Ice Extent) product available
from the National Snow and Ice Data Center:
https://nsidc.org/data/masie/browse_regions
It is given by paper J.Walsh et. al., 2019. Benchmark seasonal prediction skill estimates based on regional indices.
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
logging.basicConfig(filename = os.path.join(output_path,'logFile_train.log'),
                    filemode = 'w+', level = logging.DEBUG,
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

if __name__=="__main__":
    print ('*********************** get the key to the datasets *************************')
    # weekly variables on ERAI grid
    dataset_ERAI_fields_sic = Dataset(os.path.join(datapath,
                                      'sic_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_slp = Dataset(os.path.join(datapath_ERAI,
#                                       'slp_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_t2m = Dataset(os.path.join(datapath_ERAI,
#                                       't2m_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_z500 = Dataset(os.path.join(datapath_ERAI,
#                                        'z500_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_z850 = Dataset(os.path.join(datapath_ERAI,
#                                        'z850_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_uv10m = Dataset(os.path.join(datapath_ERAI,
#                                        'uv10m_weekly_erai_1979_2017.nc'))
#     dataset_ERAI_fields_rad = Dataset(os.path.join(datapath_ERAI,
#                                         'rad_flux_weekly_erai_1979_2017.nc'))
    #dataset_PIOMASS_siv = Dataset(os.path.join(datapath_PIOMASS,
    #                             'siv_monthly_PIOMASS_1979_2017.nc'))
    # OHC interpolated on ERA-Interim grid
    dataset_ORAS4_OHC = Dataset(os.path.join(datapath,
                                'ohc_monthly_oras2erai_1978_2017.nc'))
#     dataset_index = Dataset(os.path.join(datapath_clim_index,
#                             'index_climate_monthly_regress_1950_2017.nc'))
    #dataset_ERAI_fields_flux = Dataset(os.path.join(datapath_ERAI_fields,
    #                                  'surface_erai_monthly_regress_1979_2017_radiation.nc'))
    # mask
    dataset_ORAS4_mask = Dataset(os.path.join(datapath, 'mesh_mask.nc'))
    print ('*********************** extract variables *************************')
    #################################################################################
    #########                        data gallery                           #########
    #################################################################################
    # we use time series from 1979 to 2016 (468 months in total)
    # training data: 1979 - 2013
    # validation: 2014 - 2016
    # variables list:
    # integrals from spatial fields cover the area from 20N - 90N (4D fields [year, month, lat, lon])
    # *************************************************************************************** #
    # SIC (ERA-Interim) - benckmark
    SIC_ERAI = dataset_ERAI_fields_sic.variables['sic'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
    year_ERAI = dataset_ERAI_fields_sic.variables['year'][:-1]
    week_ERAI = dataset_ERAI_fields_sic.variables['week'][:]
    latitude_ERAI = dataset_ERAI_fields_sic.variables['latitude'][:]
    longitude_ERAI = dataset_ERAI_fields_sic.variables['longitude'][:]
    # T2M (ERA-Interim)
#     T2M_ERAI = dataset_ERAI_fields_t2m.variables['t2m'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_t2m = dataset_ERAI_fields_t2m.variables['year'][:-1]
#     week_ERAI_t2m = dataset_ERAI_fields_t2m.variables['week'][:]
#     latitude_ERAI_t2m = dataset_ERAI_fields_t2m.variables['latitude'][:]
#     longitude_ERAI_t2m = dataset_ERAI_fields_t2m.variables['longitude'][:]
    # SLP (ERA-Interim)
#     SLP_ERAI = dataset_ERAI_fields_slp.variables['slp'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_slp = dataset_ERAI_fields_slp.variables['year'][:-1]
#     week_ERAI_slp = dataset_ERAI_fields_slp.variables['week'][:]
#     latitude_ERAI_slp = dataset_ERAI_fields_slp.variables['latitude'][:]
#     longitude_ERAI_slp = dataset_ERAI_fields_slp.variables['longitude'][:]
    # Z500 (ERA-Interim)
#     Z500_ERAI = dataset_ERAI_fields_z500.variables['z'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_z500 = dataset_ERAI_fields_z500.variables['year'][:-1]
#     week_ERAI_z500 = dataset_ERAI_fields_z500.variables['week'][:]
#     latitude_ERAI_z500 = dataset_ERAI_fields_z500.variables['latitude'][:]
#     longitude_ERAI_z500 = dataset_ERAI_fields_z500.variables['longitude'][:]
    # Z850 (ERA-Interim)
#     Z850_ERAI = dataset_ERAI_fields_z850.variables['z'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_z850 = dataset_ERAI_fields_z850.variables['year'][:-1]
#     week_ERAI_z850 = dataset_ERAI_fields_z850.variables['week'][:]
#     latitude_ERAI_z850 = dataset_ERAI_fields_z850.variables['latitude'][:]
#     longitude_ERAI_z850 = dataset_ERAI_fields_z850.variables['longitude'][:]
    # UV10M (ERA-Interim)
#     U10M_ERAI = dataset_ERAI_fields_uv10m.variables['u10m'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     V10M_ERAI = dataset_ERAI_fields_uv10m.variables['v10m'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_uv10m = dataset_ERAI_fields_uv10m.variables['year'][:-1]
#     week_ERAI_uv10m = dataset_ERAI_fields_uv10m.variables['week'][:]
#     latitude_ERAI_uv10m = dataset_ERAI_fields_uv10m.variables['latitude'][:]
#     longitude_ERAI_uv10m = dataset_ERAI_fields_uv10m.variables['longitude'][:]
    # SFlux (ERA-Interim)
#     SFlux_ERAI = dataset_ERAI_fields_rad.variables['SFlux'][:-1,:,:,:] # 4D fields [year, week, lat, lon]
#     year_ERAI_SFlux = dataset_ERAI_fields_rad.variables['year'][:-1]
#     week_ERAI_SFlux = dataset_ERAI_fields_rad.variables['week'][:]
#     latitude_ERAI_SFlux = dataset_ERAI_fields_rad.variables['latitude'][:]
#     longitude_ERAI_SFlux = dataset_ERAI_fields_rad.variables['longitude'][:]
    #SIV (PIOMASS)
    #SIV_PIOMASS = dataset_PIOMASS_siv.variables['SIV'][:-12]
    #year_SIV = dataset_PIOMASS_siv.variables['year'][:-1]
    # OHC (ORAS4)
    # from 1978 - 2017 (for interpolation) / from 90 N upto 40 N
    OHC_300_ORAS4 = dataset_ORAS4_OHC.variables['OHC'][:-1,:,:67,:]/1000 # unit Peta Joule
    latitude_ORAS4 = dataset_ORAS4_OHC.variables['latitude'][:]
    longitude_ORAS4 = dataset_ORAS4_OHC.variables['longitude'][:]
    mask_OHC = np.ma.getmask(OHC_300_ORAS4[0,0,:,:])
    # AO-NAO-AMO-NINO3.4 (NOAA)
#     AO = dataset_index.variables['AO'][348:-1] # from 1979 - 2017
#     NAO = dataset_index.variables['NAO'][348:-1]
#     NINO = dataset_index.variables['NINO'][348:-1]
#     AMO = dataset_index.variables['AMO'][348:-1]
#     PDO = dataset_index.variables['PDO'][348:-1]
    logging.info("Loading datasets and extracting variables successfully!")
    #################################################################################
    ###########                 global land-sea mask                      ###########
    #################################################################################
    sea_ice_mask_global = np.ones((len(latitude_ERAI),len(longitude_ERAI)),dtype=float)
    sea_ice_mask_global[SIC_ERAI[0,0,:,:]==-1] = 0
    #################################################################################
    ###########                regionalization sea mask                   ###########
    #################################################################################
    print ('*********************** create mask *************************')
    # W:-156 E:-124 N:80 S:67
    mask_Beaufort = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:-180 E:-156 N:80 S:66
    mask_Chukchi = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:146 E:180 N:80 S:67
    mask_EastSiberian = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:100 E:146 N:80 S:67
    mask_Laptev = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:60 E:100 N:80 S:67
    mask_Kara = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:18 E:60 N:80 S:64
    mask_Barents = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:-44 E:18 N:80 S:55
    mask_Greenland = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    # W:-180 E:180 N:90 S:80
    mask_CenArctic = np.zeros((len(latitude_ERAI),len(longitude_ERAI)),dtype=int)
    print ('*********************** calc mask *************************')
    mask_Beaufort[13:31,32:76] = 1

    mask_Chukchi[13:32,0:32] = 1
    mask_Chukchi[13:32,-1] = 1

    mask_EastSiberian[13:31,434:479] = 1

    mask_Laptev[13:31,374:434] = 1

    mask_Kara[13:31,320:374] = 1

    mask_Barents[13:36,264:320] = 1

    mask_Greenland[13:47,179:264] = 1
    mask_Greenland[26:47,240:264] = 0

    mask_CenArctic[:13,:] = 1
    print ('*********************** packing *************************')
    mask_dict = {'Beaufort': mask_Beaufort[:,:],
                 'Chukchi': mask_Chukchi[:,:],
                 'EastSiberian': mask_EastSiberian[:,:],
                 'Laptev': mask_Laptev[:,:],
                 'Kara': mask_Kara[:,:],
                 'Barents': mask_Barents[:,:],
                 'Greenland': mask_Greenland[:,:],
                 'CenArctic': mask_CenArctic[:,:]}
    seas_namelist = ['Beaufort','Chukchi','EastSiberian','Laptev',
                     'Kara', 'Barents', 'Greenland','CenArctic']
    
    #################################################################################
    ########                  temporal interpolation matrix                  ########
    #################################################################################
    # interpolate from monthly to weekly
    # original monthly data will be taken as the last week of the month
    OHC_300_ORAS4_weekly_series = np.zeros(SIC_ERAI.reshape(len(year_ERAI)*48,
                                           len(latitude_ERAI),len(longitude_ERAI)).shape,
                                           dtype=float)
    OHC_300_ORAS4_series= dlacs.preprocess.operator.unfold(OHC_300_ORAS4)
    # calculate the difference between two months
    OHC_300_ORAS4_deviation_series = (OHC_300_ORAS4_series[1:,:,:] - OHC_300_ORAS4_series[:-1,:,:]) / 4
    for i in np.arange(4):
        OHC_300_ORAS4_weekly_series[3-i::4,:,:] = OHC_300_ORAS4_series[12:,:,:] - i * OHC_300_ORAS4_deviation_series[11:,:,:]
    print ('******************  calculate extent from spatial fields  *******************')
    # size of the grid box
    dx = 2 * np.pi * constant['R'] * np.cos(2 * np.pi * latitude_ERAI /
                                            360) / len(longitude_ERAI)
    dy = np.pi * constant['R'] / 480
    # calculate the sea ice area
    SIC_ERAI_area = np.zeros(SIC_ERAI.shape, dtype=float)
#     SFlux_ERAI_area = np.zeros(SFlux_ERAI.shape, dtype=float)
    for i in np.arange(len(latitude_ERAI[:])):
        # change the unit to terawatt
        SIC_ERAI_area[:,:,i,:] = SIC_ERAI[:,:,i,:]* dx[i] * dy / 1E+6 # unit km2
#         SFlux_ERAI_area[:,:,i,:] = SFlux_ERAI[:,:,i,:]* dx[i] * dy / 1E+12 # unit TeraWatt
    SIC_ERAI_area[SIC_ERAI_area<0] = 0 # switch the mask from -1 to 0
    print ('================  reshape input data into time series  =================')
    SIC_ERAI_area_series = dlacs.preprocess.operator.unfold(SIC_ERAI_area)
#     T2M_ERAI_series = dlacs.preprocess.operator.unfold(T2M_ERAI)
#     SLP_ERAI_series = dlacs.preprocess.operator.unfold(SLP_ERAI)
#     Z500_ERAI_series = dlacs.preprocess.operator.unfold(Z500_ERAI)
#     Z850_ERAI_series = dlacs.preprocess.operator.unfold(Z850_ERAI)
#     U10M_ERAI_series = dlacs.preprocess.operator.unfold(U10M_ERAI)
#     V10M_ERAI_series = dlacs.preprocess.operator.unfold(V10M_ERAI)
#     SFlux_ERAI_area_series = dlacs.preprocess.operator.unfold(SFlux_ERAI_area)
    print ('******************  choose the fields from target region  *******************')
    # select land-sea mask
    sea_ice_mask_barents = sea_ice_mask_global[12:36,264:320]
    # select the area between greenland and ice land for instance 60-70 N / 44-18 W
    sic_exp = SIC_ERAI_area_series[:,12:36,264:320]
#     t2m_exp = T2M_ERAI_series[:,12:36,264:320]
#     slp_exp = SLP_ERAI_series[:,12:36,264:320]
#     z500_exp = Z500_ERAI_series[:,12:36,264:320]
#     z850_exp = Z850_ERAI_series[:,12:36,264:320]
#     u10m_exp = U10M_ERAI_series[:,12:36,264:320]
#     v10m_exp = V10M_ERAI_series[:,12:36,264:320]
#     sflux_exp = SFlux_ERAI_area_series[:,12:36,264:320]
    ohc_exp = OHC_300_ORAS4_weekly_series[:,12:36,264:320]
    print ('*******************  pre-processing  *********************')
    print ('=========================   normalize data   ===========================')
    sic_exp_norm = dlacs.preprocess.operator.normalize(sic_exp)
#     t2m_exp_norm = deepclim.preprocess.operator.normalize(t2m_exp)
#     slp_exp_norm = deepclim.preprocess.operator.normalize(slp_exp)
#     z500_exp_norm = deepclim.preprocess.operator.normalize(z500_exp)
#     z850_exp_norm = deepclim.preprocess.operator.normalize(z850_exp)
#     u10m_exp_norm = deepclim.preprocess.operator.normalize(u10m_exp)
#     v10m_exp_norm = deepclim.preprocess.operator.normalize(v10m_exp)
#     sflux_exp_norm = deepclim.preprocess.operator.normalize(sflux_exp)
    ohc_exp_norm = dlacs.preprocess.operator.normalize(ohc_exp)
    print('================  save the normalizing factor  =================')
    sic_max = np.amax(sic_exp)
    sic_min = np.amin(sic_exp)
    print ('====================    A series of time (index)    ====================')
    _, yy, xx = sic_exp_norm.shape # get the lat lon dimension
    year = np.arange(1979,2017,1)
    year_cycle = np.repeat(year,48)
    month_cycle = np.repeat(np.arange(1,13,1),4)
    month_cycle = np.tile(month_cycle,len(year)+1) # one extra repeat for lead time dependent prediction
    month_cycle.astype(float)
    month_2D = np.repeat(month_cycle[:,np.newaxis],yy,1)
    month_exp = np.repeat(month_2D[:,:,np.newaxis],xx,2)
    print ('===================  artificial data for evaluation ====================')
    # calculate climatology of SIC
#     seansonal_cycle_SIC = np.zeros(48,dtype=float)
#     for i in np.arange(48):
#         seansonal_cycle_SIC[i] = np.mean(SIC_ERAI_sum_norm[i::48],axis=0)
    # weight for loss
#     weight_month = np.array([0,1,1,
#                              1,0,0,
#                              1,1,1,
#                              0,0,0])
    #weight_loss = np.repeat(weight_month,4)
    #weight_loss = np.tile(weight_loss,len(year))
    logging.info("Data preprocessing complete!")
    print ('*******************  parameter for check  *********************')
    choice_exp_norm = ohc_exp_norm
    print ('*******************  create basic dimensions for tensor and network  *********************')
    # specifications of neural network
    input_channels = 3
    hidden_channels = [3, 2, 1] # number of channels & hidden layers, the channels of last layer is the channels of output, too
    #hidden_channels = [3, 3, 3, 3, 2]
    #hidden_channels = [2]
    kernel_size = 3
    # here we input a sequence and predict the next step only
    #step = 1 # how many steps to predict ahead
    #effective_step = [0] # step to output
    batch_size = 1
    #num_layers = 1
    learning_rate = 0.01
    num_epochs = 1500
    print ('*******************  cross validation and testing data  *********************')
    # take 10% data as cross-validation data
    cross_valid_year = 4
    # take 10% years as testing data
    test_year = 4
    # minibatch
    #iterations = 3 # training data divided into 3 sets
    print ('*******************  check the environment  *********************')
    print ("Pytorch version {}".format(torch.__version__))
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print("Is CUDA available? {}".format(use_cuda))
    logging.info("Is CUDA available? {}".format(use_cuda))
    # CUDA settings torch.__version__ must > 0.4
    # !!! This is important for the model!!! The first option is gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##################################################################################
    ##################################################################################
    ##################################################################################
    print ('*******************  train BayesConvLSTM  *********************')
    print ('The model is designed to make many to one prediction.')
    print ('A series of multi-chanel variables will be input to the model.')
    print ('The model learns by verifying the output at each timestep.')
    # check the sequence length
    sequence_len, height, width = sic_exp_norm.shape
    # initialize our model
    model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels,
                                              kernel_size, cell_type="full").to(device)
    #model = dlacs.BayesConvLSTM.BayesConvLSTM(input_channels, hidden_channels, kernel_size).to(device)
    # use Evidence Lower Bound (ELBO) to quantify the loss
    ELBO = dlacs.function.ELBO(height*width)
    # for classification, target must be integers (label)
    #ELBO = dlacs.function.ELBO(height*width,loss_function=torch.nn.KLDivLoss())
    #ELBO = dlacs.function.ELBO(height*width,loss_function=nn.CrossEntropyLoss(reduction='mean'))
    #ELBO = dlacs.function.ELBO(height*width,loss_function=nn.NLLLoss(reduction='mean'))
    # penalty for kl
    penalty_kl = sequence_len
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
        for timestep in range(sequence_len - cross_valid_year*12*4 - test_year*12*4):
            # hidden state re-initialized inside the model when timestep=0
            #################################################################################
            ########          create input tensor with multi-input dimension         ########
            #################################################################################
            # create variables
            x_input = np.stack((sic_exp_norm[timestep,:,:],
                                choice_exp_norm[timestep,:,:],
                                month_exp[timestep,:,:])) #vstack,hstack,dstack
            x_var = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width)).to(device)
            #################################################################################
            ########       create training tensor with multi-input dimension         ########
            #################################################################################
            y_train_stack = sic_exp_norm[timestep+1,:,:] #vstack,hstack,dstack
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
            #print (timestep)
        #print(y_pred.shape)
        #print(y_train.shape)
        # print loss at certain iteration
        if t % 5 == 0:
            print("Epoch {} ELBO: {:0.3f}".format(t, loss.item()))
            logging.info("Epoch {} MSE: {:0.3f}".format(t,loss.item()))
            print("likelihood cost {:0.3f} #*# complexity cost {:0.3f}".format(likelihood.item(), complexity.item()))
            logging.info("likelihood cost {:0.3f} #*# complexity cost {:0.3f}".format(likelihood.item(), complexity.item()))
            #print(y_pred)
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

        # Update parameters
        optimiser.step()
        
    # save the model
    # (recommended) save the model parameters only
    torch.save(model.state_dict(), os.path.join(output_path,'bayesconvlstm.pkl'))
    # save the entire model
    # torch.save(model, os.path.join(output_path,'bayesconvlstm.pkl'))
    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))
    logging.info("--- %s minutes ---" % ((tttt.time() - start_time)/60))
    
    print ("*******************  Loss with time  **********************")
    fig00 = plt.figure()
    plt.plot(hist, 'r', label="Training loss")
    plt.plot(hist_likelihood, 'g', label="Likelihood loss")
    plt.plot(hist_complexity, 'b', label="Complexity loss")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    fig00.savefig(os.path.join(output_path,'SIC_ERAI_LSTM_pred_error.png'),dpi=200)