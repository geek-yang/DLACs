# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function     : Predict the Spatial Sea Ice Concentration with BayesConvLSTM at weekly time scale
Author       : Yang Liu
First Built  : 2020.03.09
Last Update  : 2020.03.09
Library      : Pytorth, Numpy, NetCDF4, os, iris, cartopy, dlacs, matplotlib
Description     : This notebook serves to predict the Arctic sea ice using deep learning. The Bayesian Convolutional Long Short Time Memory neural network is used to deal with this spatial-temporal sequence problem. We use Pytorch as the deep learning framework.

Here we predict sea ice concentration with one extra relevant field from either ocean or atmosphere to test the predictor.

Return Values   : pkl model and figures

The regionalization adopted here follows that of the MASIE (Multisensor Analyzed Sea Ice Extent) product available from the National Snow and Ice Data Center:
https://nsidc.org/data/masie/browse_regions
It is given by paper J.Walsh et. al., 2019. Benchmark seasonal prediction skill estimates based on regional indices.
"""

import sys
import warnings
import numbers

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
sys.path.append("../../../DLACs")
import dlacs
import dlacs.BayesConvLSTM
import dlacs.preprocess
import dlacs.function
