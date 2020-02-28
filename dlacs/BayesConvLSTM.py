# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Bayesian Convolutional LSTM for one step prediction
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2020.02.18
Last Update     : 2020.02.27
Description     : This module provides several methods to perform Bayesian deep learning
                  on climate data. It is designed for weather/climate prediction with 
                  spatial-temporal sequence data. The main method here is the
                  Bayesian Convolutional-Long Short Term Memory.

                  This method is devised based on the reference, namely the Bayes by Backprop:
                  Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in                       neural networks. arXiv preprint arXiv:1505.05424.
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

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesConvLSTMCell(nn.Module):
	"""
	Construction of Bayesian Convolutional LSTM Cell.
	"""
	def __init__(self, input_channels, hidden_channels, kernel_size=(1,1), alpha_shape=(1,1),
                 stride=1, padding=0, dilation=1, bias=True):
	"""
	Build Bayesian Convolutional LSTM Cell with a distribution over each gate  (weights)
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
	        create the weight, bias and kernel matrix ourself. The learnable variables are mu
            and alpha in this case, which is chosen with local reparameterization trick.
	        We are optimizing the out_bias & out_nobias matrix, which contain the weight we need.
	"""
	super(BayesConvLSTMCell, self).__init__()
	#assert hidden_channels % 2 == 0
	self.input_channels = input_channels
	self.output_channels = output_channels
	self.kernel_size = (kernel_size, kernel_size)
	self.stride = stride
    self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
	self.dilation = dilation
	self.alpha_shape = alpha_shape
	self.groups = 1 # split input into groups (input_channels) should be divisible by the number of groups'
    
    # weight/filter/kernel for the mean (mu) of each gate
    self.Wxi_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Whi_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Wxf_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Whf_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Wxc_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Whc_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Wxo_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))
    self.Who_mu = Parameter(torch.Tensor(output_channels, input_channels, *self.kernel_size))

    # register bias matrix
    if bias:
        self.Wxi_bias = Parameter(torch.Tensor(1, output_channels, 1, 1))
        self.Wxf_bias = Parameter(torch.Tensor(1, output_channels, 1, 1))
        self.Wxc_bias = Parameter(torch.Tensor(1, output_channels, 1, 1))
        self.Wxo_bias = Parameter(torch.Tensor(1, output_channels, 1, 1))
    else:
        self.register_parameter('bias', None)
    # generate the template for convolutional layer - input x filter
    # with bias
    self.Wxi_mean_out = lambda input, kernel, bias_w: F.conv2d(input, kernel, bias_w, self.stride,
                                                               self.padding, self.dilation, self.groups)
    self.Wxf_mean_out = lambda input, kernel, bias_w: F.conv2d(input, kernel, bias_w, self.stride,
                                                               self.padding, self.dilation, self.groups)
    self.Wxc_mean_out = lambda input, kernel, bias_w: F.conv2d(input, kernel, bias_w, self.stride,
                                                               self.padding, self.dilation, self.groups)
    self.Wxo_mean_out = lambda input, kernel, bias_w: F.conv2d(input, kernel, bias_w, self.stride,
                                                               self.padding, self.dilation, self.groups)
    # without bias
    self.Whi_mean_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												   self.padding, self.dilation, self.groups)
    self.Whf_mean_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												   self.padding, self.dilation, self.groups)
    self.Whc_mean_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												   self.padding, self.dilation, self.groups)
    self.Who_mean_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												 self.padding, self.dilation, self.groups)    
    # weight/filter/kernel for the variance factor (alpha) of each gate
    # in order to make sure that the variance is always positive, here we take log(alpha)
    self.Wxi_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Whi_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Wxf_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Whf_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Wxc_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Whc_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Wxo_log_alpha = Parameter(torch.Tensor(*alpha_shape))
    self.Who_log_alpha = Parameter(torch.Tensor(*alpha_shape))

    # generate the template for convolutional layer - input x filter
    # without bias
    self.Wxi_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Whi_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Wxf_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Whf_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Wxc_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Whc_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    self.Wxo_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)  
    self.Who_std_out = lambda input, kernel: F.conv2d(input, kernel, None, self.stride,
    												  self.padding, self.dilation, self.groups)
    # cell state matrix has no convolutional operation, therefore it has no distribtuion
    self.Wci = None
    self.Wcf = None
    self.Wco = None
    
    # initialize all the parameters
    self.reset_parameters()

    def reset_parameters(self):
    	"""
    	Initialise (learnable) approximate posterior parameters, including mean (mu) and 
        variance factor (log alpha).
    	"""
        n = self.input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # reset mu
        self.Wxi_mu.data.uniform_(-stdv, stdv)
        self.Whi_mu.data.uniform_(-stdv, stdv)
        self.Wxf_mu.data.uniform_(-stdv, stdv)
        self.Whf_mu.data.uniform_(-stdv, stdv)
        self.Wxc_mu.data.uniform_(-stdv, stdv)
        self.Whc_mu.data.uniform_(-stdv, stdv)
        self.Wxo_mu.data.uniform_(-stdv, stdv)
        self.Who_mu.data.uniform_(-stdv, stdv)
        # reset bias
        if self.bias is not None:
            self.Wxi_bias.data.uniform_(-stdv, stdv)
            self.Wxf_bias.data.uniform_(-stdv, stdv)
            self.Wxc_bias.data.uniform_(-stdv, stdv)
            self.Wxo_bias.data.uniform_(-stdv, stdv)
        # reset log alpha
        self.Wxi_log_alpha.data.fill_(5.0)
        self.Whi_log_alpha.data.fill_(5.0)
        self.Wxf_log_alpha.data.fill_(5.0)
        self.Whf_log_alpha.data.fill_(5.0)
        self.Wxc_log_alpha.data.fill_(5.0)
        self.Whc_log_alpha.data.fill_(5.0)
        self.Wxo_log_alpha.data.fill_(5.0)
        self.Who_log_alpha.data.fill_(5.0)

    def forward(self, x, h, c):
    	"""
    	Forward process of BayesConvLSTM Layer. This process includes two steps:
        (1) Sampling of the variational inference distribution (Gaussian)
        (2) Forward process for LSTM
    	param x: input variable
        param h: hidden state
        param c: cell state
        
        Note: the Gaussian sampling process follow the euqation (7) in
              Shridhar et. al. 2019.
    	"""
        # local parameterization trick
        # weight Wxi
    	Wxi_mean = self.Wxi_mean_out(x, self.Wxi_mu, self.Wxi_bias)
    	Wxi_sigma = torch.exp(self.Wxi_log_alpha) * self.Wxi_mu * self.Wxi_mu
    	Wxi_std = torch.sqrt(1e-16 + self.Wxi_std_out(x * x, Wxi_sigma))
    	# Sample Gaussian distribution for both training and prediction
    	# create a tensor of Gaussian noise, the same shape and data type as input tensor
        # nn.Tensor.new() Constructs a new tensor of the same data type as self tensor.
    	Wxi_epsilon = Wxi_std.data.new(Wxi_std.size()).normal_()
    	# put together the gaussian sampling result
    	Wxi = Wxi_mean + Wxi_std * Wxi_epsilon
        
        # weight Whi
        Whi_mean = self.Whi_mean_out(h, self.Whi_mu)
        Whi_sigma = torch.exp(self.Whi_log_alpha) * self.Whi_mu * self.Whi_mu
        Whi_std = torch.sqrt(1e-16 + self.Whi_std_out(h * h, Whi_sigma))
        Whi_epsilon = Whi_std.data.new(Whi_std.size()).normal_()
        Whi = Whi_mean + Whi_std * Whi_epsilon
        
        # weight Wxf
        Wxf_mean = self.Wxf_mean_out(x, self.Wxf_mu, self.Wxf_bias)
        Wxf_sigma = torch.exp(self.Wxf_log_alpha) * self.Wxf_mu * self.Wxf_mu
        Wxf_std = torch.sqrt(1e-16 + self.out_nobias(x * x, Wxf_sigma))
        Wxf_epsilon = Wxf_std.data.new(Wxf_std.size()).normal_()
        Wxf = Wxf_mean + Wxf_std * Wxf_epsilon
        
        # weight Whf
        Whf_mean = self.Whf_mean_out(h, self.Whf_mu)
        Whf_sigma = torch.exp(self.Whf_log_alpha) * self.Whf_mu * self.Whf_mu
        Whf_std = torch.sqrt(1e-16 + self.Whf_std_out(h * h, Whi_sigma))
        Whf_epsilon = Whf_std.data.new(Whf_std.size()).normal_()
        Whf = Whf_mean + Whf_std * Whf_epsilon
        
        # weight Wxc
        Wxc_mean = self.Wxc_mean_out(x, self.Wxc_mu, self.Wxc_bias)
        Wxc_sigma = torch.exp(self.Wxc_log_alpha) * self.Wxc_mu * self.Wxc_mu
        Wxc_std = torch.sqrt(1e-16 + self.Wxc_std_out(x * x, Wxc_sigma))
        Wxc_epsilon = Wxc_std.data.new(Wxc_std.size()).normal_()
        Wxc = Wxc_mean + Wxc_std * Wxc_epsilon

        # weight Whc
        Whc_mean = self.Whc_mean_out(h, self.Whc_mu)
        Whc_sigma = torch.exp(self.Whc_log_alpha) * self.Whc_mu * self.Whc_mu
        Whc_std = torch.sqrt(1e-16 + self.Whc_std_out(h * h, Whi_sigma))
        Whc_epsilon = Whc_std.data.new(Whc_std.size()).normal_()
        Whc = Whc_mean + Whc_std * Whc_epsilon

        # weight Wxo
        Wxo_mean = self.Wxo_mean_out(x, self.Wxo_mu, self.Wxo_bias)
        Wxo_sigma = torch.exp(self.Wxo_log_alpha) * self.Wxo_mu * self.Wxo_mu
        Wxo_std = torch.sqrt(1e-16 + self.Wxo_std_out(x * x, Wxo_sigma))
        Wxo_epsilon = Wxc_std.data.new(Wxo_std.size()).normal_()
        Wxo = Wxo_mean + Wxo_std * Wxo_epsilon

        # weight Whc
        Who_mean = self.Who_mean_out(h, self.Who_mu)
        Who_sigma = torch.exp(self.Who_log_alpha) * self.Who_mu * self.Who_mu
        Who_std = torch.sqrt(1e-16 + self.Who_std_out(h * h, Whi_sigma))
        Who_epsilon = Who_std.data.new(Who_std.size()).normal_()
        Who = Who_mean + Who_std * Who_epsilon
        
        ci = torch.sigmoid(Wxi + Whi + c * self.Wci)
        cf = torch.sigmoid(Wxf + Whf + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(Wxc + Whc)
        co = torch.sigmoid(Wxo + Who + cc * self.Wco)
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
        
    def kl_loss(self):
        """
        It calls the function of Kullback-Leibler divergence to compute the entropy difference.
        Since the prior distribution is prescribed as Gaussian and it contains no trainable
        parameters, we will take it as a constant and simply omit it.
        """
    	return self.weight.nelement() / self.log_alpha.nelement() * metrics.calculate_kl(self.log_alpha)



class BayesConvLSTM(nn.Module):
    """
    This is the main BayesConvLSTM module.
    param input_channels: number of channels (variables) from input fields
    param hidden_channels: number of channels inside hidden layers, for multiple layers use tuple, the dimension correponds to the output size
    param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
    param step: this parameter indicates the time step to predict ahead of the given data
    param effective_step: this parameter determines the source of the final output from the chosen step
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        #self.step = step
        #self.effective_step = effective_step
        # create a list of layers
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)        

    def forward(self, x, timestep):
        """
        Forward module of ConvLSTM.
        param x: input data with dimensions [batch size, channel, height, width]
        """
        if timestep == 0:
            self.internal_state = []
        # loop inside 
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            if timestep == 0:
                print('Initialization')
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

