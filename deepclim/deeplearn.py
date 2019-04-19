# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Deep Learning Operator for Climate Data
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.04.18
Last Update     : 2019.04.18
Contributor     :
Description     : This module provides several methods to perform deep learning
                  on climate data.
Return Values   : time series / array
Caveat!         :
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class deeplearn:
    def __init__(self,var, in_n, out_n,)