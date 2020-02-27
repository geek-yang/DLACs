# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Bayesian Convolutional LSTM for one step prediction
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2020.02.18
Last Update     : 2020.02.25
Description     : This module provides several methods to perform Bayesian deep learning
                  on climate data. It is designed for weather/climate prediction with 
                  spatial-temporal sequence data. The main method here is the
                  Bayesian Convolutional-Long Short Term Memory.

                  This method is devised based on the reference:
                  Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424.
Return Values   : time series / array
Caveat!         : This module get input as a spatial-temporal sequence and make a prediction for only one step!!
				  The so-called many to one prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class BayesConvCell(nn.Module):
	"""
	Construction of Bayesian Convolutional LSTM Cell.
	"""
	def __init__(self, input_channels, hidden_channels, kernel_size, alpha_shape=(1,1), stride=1,
                 padding=0, dilation=1, bias=True):
	"""
	Build Bayesian Convolutional Cell with a distribution over each of the weights and biases
    in the layer. The cell is designed to enable the implementation of back-propagation process.
	param input_channels: number of channels (variables) from input fields
	param hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
	param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
	param alpha_shape: the scalar multiplier
	param stride: number of pixels by which the window moves after each operation
	param padding: number of pixels to preserve the input shape and information from edge
	param dilation: controls the spacing between the kernel points
	param bias: an additive bias

	Caveat: This module is customized using basic modules from pytorch. Therefore, we need to
	        create the weight, bias and kernel matrix ourself.
	        We are optimizing the out_bias & out_nobias matrix, which are actually the weight we need.
	"""
	super(BayesConvCell, self).__init__()
	#assert hidden_channels % 2 == 0
	self.input_channels = input_channels
	self.output_channels = output_channels
	self.kernel_size = (kernel_size, kernel_size)
	self.stride = stride
	self.padding = padding
	self.dilation = dilation
	self.alpha_shape = alpha_shape
	self.groups = 1 # split input into groups (input_channels) should be divisible by the number of groups
    # weight/filter/kernel
    self.weight = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    if bias:
        self.bias = Parameter(torch.Tensor(1, output_channels, 1, 1))
    else:
        self.register_parameter('bias', None)
    # convolutional layer - input x filter
    self.out_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride,
    											   self.padding, self.dilation, self.groups)
    self.out_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												 self.padding, self.dilation, self.groups)
    self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.reset_parameters()

    def reset_parameters(self):
    	"""
    	Initialise (learnable) approximate posterior parameters, including bias and weights.
    	"""
        n = self.input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):
    	"""
    	Forward process of BayesConv Layer.
    	param x: input variable
    	"""
    	# local parameterization trick
    	mean = self.out_bias(x, self.weight)
    	sigma = torch.exp(self.log_alpha) * self.weight * self.weight
    	std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
    	# Sample Gaussian distribution for training for not for prediction
    	if self.training:
    		# create a tensor of Gaussian noise, the same shape and data type as input tensor
    		epsilon = std.data.new(std.size()).normal_()
    	else:
    		epsilon = 0.0
    	# Local reparameterization trick
    	out = mean + std * epsilon

    	return return out


    	ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)

    def kl_loss(self):
    	return self.weight.nelement() / self.log_alpha.nelement() * metrics.calculate_kl(self.log_alpha)


class BayesConvLSTM



