# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Bayesian Convolutional LSTM for sequence to one prediction
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2020.02.18
Last Update     : 2020.10.21
Description     : This module provides several methods to perform Bayesian deep learning
                  on climate data. It is designed for weather/climate prediction with 
                  spatial-temporal sequence data. 
                  
                  The deep neural network used here is the Bayesian Convolutional-Long Short
                  Term Memory network, which is constructed with vatiaitional inference and
                  trained with the Bayes by Backprop. The local reparameterization trick is
                  applied for the back-propagation of stochastic node.

                  This network is devised based on the reference, namely :
                  Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty
                  in neural networks. arXiv preprint arXiv:1505.05424.
                  
                  Shridhar, K., Laumann, F. and Liwicki, M., 2019. A comprehensive guide to bayesian
                  convolutional neural network with variational inference. arXiv preprint arXiv:1901.02731.
                  
                  Fortunato, M., Blundell, C. and Vinyals, O., 2017. Bayesian recurrent neural networks.
                  arXiv preprint arXiv:1704.02798.
                  
Return Values   : time series / array
Caveat!         : This module get input as a spatial-temporal sequence and make a prediction for only
                  one step!! The so-called many to one (or sequence to one) prediction.
"""

#import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import dlacs.function

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesConvLSTMCell(nn.Module):
    """
    Construction of Bayesian Convolutional LSTM Cell.
    This is a Bayesian Convolutional LSTM Cell with local reparameterization trick 
    designed by Shridhar, K., Laumann, F. and Liwicki, M., 2019. The variance of 
    Gaussian distribution is modelled with a factor and the square of mean, which is
    enabled for back-propagation.
    Since the variance is not an independent variable, we name it as "reduced" cell.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, weight_dict = None,
                 cell_index = None, alpha_shape=(1,1), stride=1, padding=0, dilation=1, bias=True):
        """
        Build Bayesian Convolutional LSTM Cell with a distribution over each weight matrix
        in the layer. The cell is designed to enable the implementation of back-propagation process.
        
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, the
        dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        param weight_dict: weight matrix for the initialization of mu (mean)
        param cell_index: index of created BayesConvLSTM cell, for initialization with given weights
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
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1 # split input into groups (input_channels) should be divisible by the number of groups'
        self.bias = bias
        
        # inherit weight matrix for mu
        # this used to initialize BayesConvLSTM with ConvLSTM
        self.weight_dict = weight_dict
        self.cell_index = cell_index

        # weight/filter/kernel for the mean (mu) of each gate
        self.Wxi_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whi_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxf_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whf_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxc_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whc_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxo_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Who_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))

        # register bias matrix
        if self.bias:
            self.Wxi_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxf_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxc_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxo_bias = Parameter(torch.Tensor(hidden_channels))
        else:
            self.register_parameter('bias', None) # method from nn, since bias is learnable parameter
        # generate the convolutional layer for mean - input x filter
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
        # About its definition, see: 
        # Shridhar, K., Laumann, F. and Liwicki, M., 2019. A comprehensive guide to bayesian
        # convolutional neural network with variational inference. arXiv preprint arXiv:1901.02731.
        self.Wxi_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Whi_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Wxf_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Whf_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Wxc_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Whc_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Wxo_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.Who_log_alpha = Parameter(torch.Tensor(*alpha_shape))
        # generate the convolutional layer for standard deviation - input x filter
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
        Note: Here we follow the variational inference and take a Gaussian distribution
              as N(mu,alpha x mu^2). Therefore alpha is smaller than 1 and log_alpha is
              smaller than 0.
        """
        if self.weight_dict is None:
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
        else:
            self.Wxi_mu.data = self.weight_dict['cell{}.Wxi.weight'.format(self.cell_index)].data
            self.Whi_mu.data = self.weight_dict['cell{}.Whi.weight'.format(self.cell_index)].data
            self.Wxf_mu.data = self.weight_dict['cell{}.Wxf.weight'.format(self.cell_index)].data
            self.Whf_mu.data = self.weight_dict['cell{}.Whf.weight'.format(self.cell_index)].data
            self.Wxc_mu.data = self.weight_dict['cell{}.Wxc.weight'.format(self.cell_index)].data
            self.Whc_mu.data = self.weight_dict['cell{}.Whc.weight'.format(self.cell_index)].data
            self.Wxo_mu.data = self.weight_dict['cell{}.Wxo.weight'.format(self.cell_index)].data
            self.Who_mu.data = self.weight_dict['cell{}.Who.weight'.format(self.cell_index)].data
            # reset bias
            if self.bias is not None:
                self.Wxi_bias.data = self.weight_dict['cell{}.Wxi.bias'.format(self.cell_index)].data
                self.Wxf_bias.data = self.weight_dict['cell{}.Wxf.bias'.format(self.cell_index)].data
                self.Wxc_bias.data = self.weight_dict['cell{}.Wxc.bias'.format(self.cell_index)].data
                self.Wxo_bias.data = self.weight_dict['cell{}.Wxo.bias'.format(self.cell_index)].data
                
        # reset log alpha
        self.Wxi_log_alpha.data.fill_(-5.0)
        self.Whi_log_alpha.data.fill_(-5.0)
        self.Wxf_log_alpha.data.fill_(-5.0)
        self.Whf_log_alpha.data.fill_(-5.0)
        self.Wxc_log_alpha.data.fill_(-5.0)
        self.Whc_log_alpha.data.fill_(-5.0)
        self.Wxo_log_alpha.data.fill_(-5.0)
        self.Who_log_alpha.data.fill_(-5.0)
        
    def forward(self, x, h, c, training=True):
        """
        Forward process of BayesConvLSTM Layer. This process includes two steps:
        (1) Sampling of the variational inference distribution (Gaussian)
        (2) Forward process for LSTM
        param x: input variable
        param h: hidden state
        param c: cell state
        param training: determine whether it in training mode or prediction mode.
                        options are "True" and "False".

        Note: the Gaussian sampling process follow the euqation (7) in
        Shridhar et. al. 2019.
        """
        # local parameterization trick
        # weight Wxi
        Wxi_mean = self.Wxi_mean_out(x, self.Wxi_mu, self.Wxi_bias)
        Wxi_var = torch.exp(self.Wxi_log_alpha) * self.Wxi_mu * self.Wxi_mu
        Wxi_std = torch.sqrt(1e-16 + self.Wxi_std_out(x * x, Wxi_var))
        # Sample Gaussian distribution for both training and prediction
        # create a tensor of Gaussian noise, the same shape and data type as input tensor
        # nn.Tensor.new() Constructs a new tensor of the same data type as self tensor.
        Wxi_epsilon = Wxi_std.data.new(Wxi_std.size()).normal_()
        # put together the gaussian sampling result
        Wxi = Wxi_mean + Wxi_std * Wxi_epsilon

        # weight Whi
        Whi_mean = self.Whi_mean_out(h, self.Whi_mu)
        Whi_var = torch.exp(self.Whi_log_alpha) * self.Whi_mu * self.Whi_mu
        Whi_std = torch.sqrt(1e-16 + self.Whi_std_out(h * h, Whi_var))
        Whi_epsilon = Whi_std.data.new(Whi_std.size()).normal_()
        Whi = Whi_mean + Whi_std * Whi_epsilon
        
        # weight Wxf
        Wxf_mean = self.Wxf_mean_out(x, self.Wxf_mu, self.Wxf_bias)
        Wxf_var = torch.exp(self.Wxf_log_alpha) * self.Wxf_mu * self.Wxf_mu
        Wxf_std = torch.sqrt(1e-16 + self.Wxf_std_out(x * x, Wxf_var))
        Wxf_epsilon = Wxf_std.data.new(Wxf_std.size()).normal_()
        Wxf = Wxf_mean + Wxf_std * Wxf_epsilon
        
        # weight Whf
        Whf_mean = self.Whf_mean_out(h, self.Whf_mu)
        Whf_var = torch.exp(self.Whf_log_alpha) * self.Whf_mu * self.Whf_mu
        Whf_std = torch.sqrt(1e-16 + self.Whf_std_out(h * h, Whf_var))
        Whf_epsilon = Whf_std.data.new(Whf_std.size()).normal_()
        Whf = Whf_mean + Whf_std * Whf_epsilon
        
        # weight Wxc
        Wxc_mean = self.Wxc_mean_out(x, self.Wxc_mu, self.Wxc_bias)
        Wxc_var = torch.exp(self.Wxc_log_alpha) * self.Wxc_mu * self.Wxc_mu
        Wxc_std = torch.sqrt(1e-16 + self.Wxc_std_out(x * x, Wxc_var))
        Wxc_epsilon = Wxc_std.data.new(Wxc_std.size()).normal_()
        Wxc = Wxc_mean + Wxc_std * Wxc_epsilon

        # weight Whc
        Whc_mean = self.Whc_mean_out(h, self.Whc_mu)
        Whc_var = torch.exp(self.Whc_log_alpha) * self.Whc_mu * self.Whc_mu
        Whc_std = torch.sqrt(1e-16 + self.Whc_std_out(h * h, Whc_var))
        Whc_epsilon = Whc_std.data.new(Whc_std.size()).normal_()
        Whc = Whc_mean + Whc_std * Whc_epsilon

        # weight Wxo
        Wxo_mean = self.Wxo_mean_out(x, self.Wxo_mu, self.Wxo_bias)
        Wxo_var = torch.exp(self.Wxo_log_alpha) * self.Wxo_mu * self.Wxo_mu
        Wxo_std = torch.sqrt(1e-16 + self.Wxo_std_out(x * x, Wxo_var))
        Wxo_epsilon = Wxc_std.data.new(Wxo_std.size()).normal_()
        Wxo = Wxo_mean + Wxo_std * Wxo_epsilon

        # weight Whc
        Who_mean = self.Who_mean_out(h, self.Who_mu)
        Who_var = torch.exp(self.Who_log_alpha) * self.Who_mu * self.Who_mu
        Who_std = torch.sqrt(1e-16 + self.Who_std_out(h * h, Who_var))
        Who_epsilon = Who_std.data.new(Who_std.size()).normal_()
        Who = Who_mean + Who_std * Who_epsilon
        
        ci = torch.sigmoid(Wxi + Whi + c * self.Wci)
        cf = torch.sigmoid(Wxf + Whf + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(Wxc + Whc)
        co = torch.sigmoid(Wxo + Who + cc * self.Wco)
        ch = co * torch.tanh(cc)

        # compute Kullback-Leibler divergence
        if training:
            Wxi_kl_loss = self.kl_loss(Wxi, Wxi_mean, Wxi_std)
            Whi_kl_loss = self.kl_loss(Whi, Whi_mean, Whi_std)
            Wxf_kl_loss = self.kl_loss(Wxf, Wxf_mean, Wxf_std)
            Whf_kl_loss = self.kl_loss(Whf, Whf_mean, Whf_std)
            Wxc_kl_loss = self.kl_loss(Wxc, Wxc_mean, Wxc_std)
            Whc_kl_loss = self.kl_loss(Whc, Whc_mean, Whc_std)
            Wxo_kl_loss = self.kl_loss(Wxo, Wxo_mean, Wxo_std)
            Who_kl_loss = self.kl_loss(Who, Who_mean, Who_std)

            kl_loss_sum = Wxi_kl_loss + Whi_kl_loss + Wxf_kl_loss + Wxf_kl_loss +\
                        Wxc_kl_loss + Whc_kl_loss + Wxo_kl_loss + Who_kl_loss
        else:
            kl_loss_sum = None
        
        return ch, cc, kl_loss_sum

    def init_hidden(self, batch_size, hidden, shape):
        """
        Initialize the hidden layers (cell state) in LSTM.
        """
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device),
                Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device))        
        
    def kl_loss(self, weight, mu, std):
        """
        This module takes Kullback-Leibler divergence to compute the entropy difference.
        It includes variational posterior (Gaussian distribution) and prior (fixed Gaussian). 
        The prior contains no trainable parameters. So we use fixed normal distribution.
        param weight: weight matrix after sampling the Gaussian
        param mu: mean of the variational inference distribution
        param std: standard deviation of the variational inference distribution
        """
        posterior_entropy = torch.sum(dlacs.function.logpdf_Gaussian(weight, mu, std))
        prior_entropy = torch.sum(dlacs.function.logpdf_Gaussian(weight))
        return posterior_entropy - prior_entropy

class BayesConvLSTMCell_F(nn.Module):
    """
    Construction of Bayesian Convolutional LSTM Cell with experimental settings.
    Caveat! This class is kept for integrating new features to the BayesConvLSTM
    cell and therefore it is only meant for testing!! In case you want to use the existing
    setup from Shridhar et. al. 2019, use the class "BayesConvLSTMCell".
    
    Current setup:
    This is a Bayesian Convolutional LSTM Cell defiend with the variance and mean of 
    Gaussian distribution represented by two independent weight matrix. It is different
    from the configuration introduced by Shridhar et. al. 2019.
    
    Since the variance is an independent variable in this case, we name it as "full" cell.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, weight_dict = None,
                 cell_index = None, stride=1, padding=0, dilation=1, bias=True):
        """
        Build Bayesian Convolutional LSTM Cell with a distribution over each gate (weights)
        in the layer. The cell is designed to enable the implementation of back-propagation process.
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        param weight_dict: weight matrix for the initialization of mu (mean)
        param cell_index: index of created BayesConvLSTM cell
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
        super(BayesConvLSTMCell_F, self).__init__()
        #assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
        self.dilation = dilation
        #self.alpha_shape = alpha_shape
        self.groups = 1 # split input into groups (input_channels) should be divisible by the number of groups'
        self.bias = bias
        
        # inherit weight matrix for mu
        self.weight_dict = weight_dict
        self.cell_index = cell_index

        # weight/filter/kernel for the mean (mu) of each gate
        self.Wxi_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whi_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxf_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whf_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxc_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whc_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxo_mu = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Who_mu = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))

        # register bias matrix
        if self.bias:
            self.Wxi_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxf_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxc_bias = Parameter(torch.Tensor(hidden_channels))
            self.Wxo_bias = Parameter(torch.Tensor(hidden_channels))
        else:
            self.register_parameter('bias', None)
        # generate the convolutional layer for mean - input x filter
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
        # in order to make sure that the variance is always positive, here we take log(var)
        self.Wxi_log_var = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whi_log_var = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxf_log_var = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whf_log_var = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxc_log_var = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Whc_log_var = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        self.Wxo_log_var = Parameter(torch.Tensor(hidden_channels, input_channels, *self.kernel_size))
        self.Who_log_var = Parameter(torch.Tensor(hidden_channels, hidden_channels, *self.kernel_size))
        # generate the convolutional layer for standard deviation - input x filter
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
        Note: Here we follow the variational inference and take a Gaussian distribution
              as N(mu,alpha x mu^2). Therefore alpha is smaller than 1 and log_alpha is
              smaller than 0.
        """
        if self.weight_dict is None:
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
        else:
            self.Wxi_mu.data = self.weight_dict['cell{}.Wxi.weight'.format(self.cell_index)].data
            self.Whi_mu.data = self.weight_dict['cell{}.Whi.weight'.format(self.cell_index)].data
            self.Wxf_mu.data = self.weight_dict['cell{}.Wxf.weight'.format(self.cell_index)].data
            self.Whf_mu.data = self.weight_dict['cell{}.Whf.weight'.format(self.cell_index)].data
            self.Wxc_mu.data = self.weight_dict['cell{}.Wxc.weight'.format(self.cell_index)].data
            self.Whc_mu.data = self.weight_dict['cell{}.Whc.weight'.format(self.cell_index)].data
            self.Wxo_mu.data = self.weight_dict['cell{}.Wxo.weight'.format(self.cell_index)].data
            self.Who_mu.data = self.weight_dict['cell{}.Who.weight'.format(self.cell_index)].data
            # reset bias
            if self.bias is not None:
                self.Wxi_bias.data = self.weight_dict['cell{}.Wxi.bias'.format(self.cell_index)].data
                self.Wxf_bias.data = self.weight_dict['cell{}.Wxf.bias'.format(self.cell_index)].data
                self.Wxc_bias.data = self.weight_dict['cell{}.Wxc.bias'.format(self.cell_index)].data
                self.Wxo_bias.data = self.weight_dict['cell{}.Wxo.bias'.format(self.cell_index)].data
        # reset log var
        self.Wxi_log_var.data.fill_(-10.0)
        self.Whi_log_var.data.fill_(-10.0)
        self.Wxf_log_var.data.fill_(-10.0)
        self.Whf_log_var.data.fill_(-10.0)
        self.Wxc_log_var.data.fill_(-10.0)
        self.Whc_log_var.data.fill_(-10.0)
        self.Wxo_log_var.data.fill_(-10.0)
        self.Who_log_var.data.fill_(-10.0)
        
    def forward(self, x, h, c, training=True):
        """
        Forward process of BayesConvLSTM Layer. This process includes two steps:
        (1) Sampling of the variational inference distribution (Gaussian)
        (2) Forward process for LSTM
        param x: input variable
        param h: hidden state
        param c: cell state

        Note: the Gaussian sampling process is DIFFERENT! from the euqation (7) in
        Shridhar et. al. 2019.
        """
        # local parameterization trick
        # weight Wxi
        Wxi_mean = self.Wxi_mean_out(x, self.Wxi_mu, self.Wxi_bias)
        Wxi_var = torch.exp(self.Wxi_log_var)
        Wxi_std = torch.sqrt(1e-16 + self.Wxi_std_out(x * x, Wxi_var))
        # Sample Gaussian distribution for both training and prediction
        # create a tensor of Gaussian noise, the same shape and data type as input tensor
        # nn.Tensor.new() Constructs a new tensor of the same data type as self tensor.
        Wxi_epsilon = Wxi_std.data.new(Wxi_std.size()).normal_()
        # put together the gaussian sampling result
        Wxi = Wxi_mean + Wxi_std * Wxi_epsilon

        # weight Whi
        Whi_mean = self.Whi_mean_out(h, self.Whi_mu)
        Whi_var = torch.exp(self.Whi_log_var)
        Whi_std = torch.sqrt(1e-16 + self.Whi_std_out(h * h, Whi_var))
        Whi_epsilon = Whi_std.data.new(Whi_std.size()).normal_()
        Whi = Whi_mean + Whi_std * Whi_epsilon
        
        # weight Wxf
        Wxf_mean = self.Wxf_mean_out(x, self.Wxf_mu, self.Wxf_bias)
        Wxf_var = torch.exp(self.Wxf_log_var)
        Wxf_std = torch.sqrt(1e-16 + self.Wxf_std_out(x * x, Wxf_var))
        Wxf_epsilon = Wxf_std.data.new(Wxf_std.size()).normal_()
        Wxf = Wxf_mean + Wxf_std * Wxf_epsilon
        
        # weight Whf
        Whf_mean = self.Whf_mean_out(h, self.Whf_mu)
        Whf_var = torch.exp(self.Whf_log_var)
        Whf_std = torch.sqrt(1e-16 + self.Whf_std_out(h * h, Whf_var))
        Whf_epsilon = Whf_std.data.new(Whf_std.size()).normal_()
        Whf = Whf_mean + Whf_std * Whf_epsilon
        
        # weight Wxc
        Wxc_mean = self.Wxc_mean_out(x, self.Wxc_mu, self.Wxc_bias)
        Wxc_var = torch.exp(self.Wxc_log_var)
        Wxc_std = torch.sqrt(1e-16 + self.Wxc_std_out(x * x, Wxc_var))
        Wxc_epsilon = Wxc_std.data.new(Wxc_std.size()).normal_()
        Wxc = Wxc_mean + Wxc_std * Wxc_epsilon

        # weight Whc
        Whc_mean = self.Whc_mean_out(h, self.Whc_mu)
        Whc_var = torch.exp(self.Whc_log_var)
        Whc_std = torch.sqrt(1e-16 + self.Whc_std_out(h * h, Whc_var))
        Whc_epsilon = Whc_std.data.new(Whc_std.size()).normal_()
        Whc = Whc_mean + Whc_std * Whc_epsilon

        # weight Wxo
        Wxo_mean = self.Wxo_mean_out(x, self.Wxo_mu, self.Wxo_bias)
        Wxo_var = torch.exp(self.Wxo_log_var)
        Wxo_std = torch.sqrt(1e-16 + self.Wxo_std_out(x * x, Wxo_var))
        Wxo_epsilon = Wxc_std.data.new(Wxo_std.size()).normal_()
        Wxo = Wxo_mean + Wxo_std * Wxo_epsilon

        # weight Whc
        Who_mean = self.Who_mean_out(h, self.Who_mu)
        Who_var = torch.exp(self.Who_log_var)
        Who_std = torch.sqrt(1e-16 + self.Who_std_out(h * h, Who_var))
        Who_epsilon = Who_std.data.new(Who_std.size()).normal_()
        Who = Who_mean + Who_std * Who_epsilon
        
        ci = torch.sigmoid(Wxi + Whi + c * self.Wci)
        cf = torch.sigmoid(Wxf + Whf + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(Wxc + Whc)
        co = torch.sigmoid(Wxo + Who + cc * self.Wco)
        ch = co * torch.tanh(cc)

        # compute Kullback-Leibler divergence
        if training:
            Wxi_kl_loss = self.kl_loss(Wxi, Wxi_mean, Wxi_std)
            Whi_kl_loss = self.kl_loss(Whi, Whi_mean, Whi_std)
            Wxf_kl_loss = self.kl_loss(Wxf, Wxf_mean, Wxf_std)
            Whf_kl_loss = self.kl_loss(Whf, Whf_mean, Whf_std)
            Wxc_kl_loss = self.kl_loss(Wxc, Wxc_mean, Wxc_std)
            Whc_kl_loss = self.kl_loss(Whc, Whc_mean, Whc_std)
            Wxo_kl_loss = self.kl_loss(Wxo, Wxo_mean, Wxo_std)
            Who_kl_loss = self.kl_loss(Who, Who_mean, Who_std)

            kl_loss_sum = Wxi_kl_loss + Whi_kl_loss + Wxf_kl_loss + Wxf_kl_loss +\
                        Wxc_kl_loss + Whc_kl_loss + Wxo_kl_loss + Who_kl_loss
        else:
            kl_loss_sum = None
        
        return ch, cc, kl_loss_sum

    def init_hidden(self, batch_size, hidden, shape):
        """
        Initialize the hidden layers (cell state) in LSTM.
        """
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device),
                Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device))        
        
    def kl_loss(self, weight, mu, std):
        """
        This module takes Kullback-Leibler divergence to compute the entropy difference.
        It includes variational posterior (Gaussian distribution) and prior (fixed Gaussian). 
        The prior contains no trainable parameters. So we use fixed normal distribution.
        param weight: weight matrix after sampling the Gaussian
        param mu: mean of the variational inference distribution
        param std: standard deviation of the variational inference distribution
        """
        posterior_entropy = torch.sum(dlacs.function.logpdf_Gaussian(weight, mu, std))
        prior_entropy = torch.sum(dlacs.function.logpdf_Gaussian(weight))
        return posterior_entropy - prior_entropy    
    
class BayesConvLSTM(nn.Module):
    """
    This is the main BayesConvLSTM module. It serves to construct the BayesConvLSTM
    with BayesConvLSTM cells defined in this module.
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, cell_type='reduced', weight_dict = None):
        """
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, for multiple layers use tuple, the dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        param step: this parameter indicates the time step to predict ahead of the given data
        param effective_step: this parameter determines the source of the final output from the chosen step
        param cell_type: determines the type of cell to be used by the neural network, options are
                         "reduced" and "full", in terms of the definition for variance in Gaussian.
        param weight_dict: weight matrix of trained models
        """
        super(BayesConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        #self.step = step
        #self.effective_step = effective_step
        print("!@#$% The network will be built with {} size BayesConvLSTM cell. !@#$%".format(cell_type))
        # create a list of layers
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if cell_type == "reduced":
                if weight_dict is None:
                    cell = BayesConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
                else:
                    #initialize model with given weight matrix (only for the mean of each weight distribution)
                    cell = BayesConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size,
                                             weight_dict, i)
            elif cell_type == "full":
                if weight_dict is None:
                    cell = BayesConvLSTMCell_F(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
                else:
                    cell = BayesConvLSTMCell_F(self.input_channels[i], self.hidden_channels[i], self.kernel_size,
                                               weight_dict, i)
            else:
                raise IOError("Please choose the right type of BayesConvLSTM cell. Check the documentation for more information.")
            setattr(self, name, cell)
            self._all_layers.append(cell)        

    def forward(self, x, timestep, training=True):
        """
        Forward module of ConvLSTM. The computation of KL-divergence in each layer is performed
        and the results will be aggregated.
        param x: input data with dimensions [batch size, channel, height, width]
        param timestep: tell the module what is the current time step.
                        the module only performs the initialization for the first timestep
        param training: determine whether it in training mode or prediction mode.
                        options are "True" and "False".
        """
        # define the type of forward
        self.training = training # training is boolean
        # define the kl loss
        kl_loss = torch.zeros(1).to(device)
        if timestep == 0:
            self.internal_state = []
        # loop inside 
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            if timestep == 0:
                #print('Initialization layer {}'.format(i))
                bsize, _, height, width = x.size()
                # initialize hidden layers (cell state) matrix
                (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                         shape=(height, width))
                self.internal_state.append((h, c))                
            # do forward
            (h, c) = self.internal_state[i]
            x, new_c, kl_loss_layer = getattr(self, name)(x, h, c, self.training)
            
            self.internal_state[i] = (x, new_c)
            
            # take KL divergence
            if training:
                kl_loss += (kl_loss_layer / self.hidden_channels[i]) # scale kl loss w.r.t. hidden channels
            
            # only record output from last layer
            if i == (self.num_layers - 1):
                outputs = x

        return outputs, kl_loss, (x, new_c)