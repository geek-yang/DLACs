# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Bayesian Convolutional LSTM with Bernoulli approximation variational inference for one step prediction
Author          : Yang Liu (y.liu@esciencecenter.nl) and Tianyi Zhang
First Built     : 2020.06.19
Last Update     : 2020.06.19
Description     : This module serves to perform Bayesian deep learning with variational inference. The variational
              inference is realized through the implementation of dropout, thus follows the Bernoulli distribution.
              It is built ontop of Convolutional-Long Short Term Memory networks and designed for emotion recognition

              This method is devised based on the reference:
              Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015).
              Convolutional LSTM network: A machine learning approach for precipitation nowcasting.
              In Advances in neural information processing systems (pp. 802-810).
              
              Gal, Y., & Ghahramani, Z. (2015). Bayesian convolutional neural networks with Bernoulli
              approximate variational inference. arXiv preprint arXiv:1506.02158.

Return Values   : time series / array
Caveat!        : This module get input as a spatial-temporal sequence and make a prediction for only one step!!
             The so-called many to one prediction.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import dlacs.function

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesConvLSTMBinaryCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, p=0.5):
        """
        Build convolutional cell for BayesConvLSTM with a Bernoulli distribution.
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        param p: probability of an element to be zero-ed (dropout)
        """
        super(BayesConvLSTMBinaryCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        #self.num_features = 4

        self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
        # input shape of nn.Conv2d (input_channels,out_channels,kernel_size, stride, padding)
        # kernal_size and stride can be tuples, indicating non-square filter / uneven stride
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        
        # probability of dropout
        self.dropout = nn.Dropout2d(p=p)
        
    def forward(self, x, h, c):
        ci = torch.sigmoid(self.dropout(self.Wxi(x)) + self.dropout(self.Whi(h)) + c * self.Wci)
        cf = torch.sigmoid(self.dropout(self.Wxf(x)) + self.dropout(self.Whf(h)) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.dropout(self.Wxc(x)) + self.dropout(self.Whc(h)))
        co = torch.sigmoid(self.dropout(self.Wxo(x)) + self.dropout(self.Who(h)) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device),
                Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device))

class BayesConvLSTMBinary(nn.Module):
    """
    This is the main BayesConvLSTMBinary module.
    param input_channels: number of channels (variables) from input fields
    param hidden_channels: number of channels inside hidden layers, for multiple layers use tuple, the dimension correponds to the output size
    param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
    param step: this parameter indicates the time step to predict ahead of the given data
    param effective_step: this parameter determines the source of the final output from the chosen step
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, p=0.5):
        super(BayesConvLSTMBinary, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        #self.step = step
        #self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = BayesConvLSTMBinaryCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, p)
            setattr(self, name, cell)
            self._all_layers.append(cell)        

    def forward(self, x, timestep):
        """
        Forward module of BayesConvLSTMBinary.
        param x: input data with dimensions [batch size, channel, height, width]
        """
        if timestep == 0:
            self.internal_state = []
        # loop inside
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            if timestep == 0:
                #print('Initialization')
                bsize, _, height, width = x.size()
                (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                         shape=(height, width))
                self.internal_state.append((h, c))
                    
            # do forward
            (h, c) = self.internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            self.internal_state[i] = (x, new_c)
            # only record output from last layer
            if i == (self.num_layers - 1):
                outputs = x

        return outputs, (x, new_c)
