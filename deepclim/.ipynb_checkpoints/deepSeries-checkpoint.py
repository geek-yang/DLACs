# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Deep Learning Operator for Climate Data
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.04.18
Last Update     : 2019.05.08
Contributor     :
Description     : This module provides several methods to perform deep learning
                  on climate data. It is designed for time series prediction with
                  temporal sequence data.
Return Values   : time series / array
Caveat!         :
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=1):
        """
        Initialize the LSTM module in Pytorch and specify the basic model structure.
        param input_dim: dimension of input array
        """
        super(LSTM, self).__init__() #inheritance https://realpython.com/python-super/
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        #self.lstm2 = nn.LSTM(self.hidden_size,self.hidden_size // 4,1,dropout=0.1,bidirectional=self.bi-1,batch_first=True)

        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
    
    def init_hidden(self):
        """
        Initialization of hidden state. We create two matrix: hidden and cell.
        The shape of the matrix should be: 
        torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        """
        
        return (torch.randn(self.num_layers, self.batch_size, self.hidden_dim),
                torch.randn(self.num_layers, self.batch_size, self.hidden_dim))

#     def init_hidden2(self):
#         """
#         Initialization of hidden state. We create two matrix: hidden and cell.
#         The shape of the matrix should be: 
#         torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
#         """
        
#         return (torch.randn(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.randn(self.num_layers, self.batch_size, self.hidden_dim))
    
    def forward(self, x_input):
        """
        Forward pass through LSTM layer.
        The shape of lstm_out: [input_size, batch_size, hidden_dim]
        The shape of self.hidden: (hn, cn), where hn and cn both
        have shape (num_layers, batch_size, hidden_dim).
        More information is available:
        https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
        """
        # matrix dimension as feed for lstm is fixed
        # torch.tensor.view(-1) adpat the array to a new shape with one dimension unknown [..,-1,..]
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.size(0), self.batch_size, -1))
        #lstm_out2, self.hidden2 = self.lstm(lstm_out)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction!!!
        # F.linear is able to handle multi-dimensional matrix, but in case...
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out.view(lstm_out.size(0)*self.batch_size, -1))
        #return y_pred.view(-1) # tensor.view only rearrange the array, not transpose
        return y_pred.view(lstm_out.size(0),self.batch_size,-1)
        
class LinearReg(torch.nn.Module):
    def __init__(self,input_dim,):
        """
        Initialize the Linear Regression module in Pytorch and specify the basic model structure.       
        """
        super(LinearReg, self).__init__() #inheritance https://realpython.com/python-super/
        self.linear = torch.nn.Linear(input_dim,output_dim=1)
        
    def forward(self, x_input):
        """
        Forward training data to the model.
        """
        y_output = self.linear(x_input)
        
        return y_output