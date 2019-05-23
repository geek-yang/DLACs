# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Convolutional LSTM for multiple steps prediction
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.05.21
Last Update     : 2019.05.21
Contributor     :
Description     : This module provides several methods to perform deep learning
                  on climate data. It is designed for weather/climate prediction with
                  spatial-temporal sequence data. The main method here is the 
                  Convolutional-Long Short Term Memory, which is first used by
                  Shi et. al. (2015) for the prediction of precipitation. There paper
                  is available through the link:
                  Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
                  https://arxiv.org/abs/1506.04214
                  The module is designed with reference to the script:
                  https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
Return Values   : time series / array
Caveat!         : The variables are not generated for the usa of cuda. The original code is cuda specified.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """
        Build convolutional cell for ConvLSTM.
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        """
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

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

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.randn(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.randn(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    """
    This is the main ConvLSTM module.
    param input_channels: number of channels (variables) from input fields
    param hidden_channels: number of channels inside hidden layers, for multiple layers use tuple, the dimension correponds to the output size
    param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
    param step: this parameter indicates the time step to predict ahead of the given data
    param effective_step: this parameter determines the source of the final output from the chosen step
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        """
        Forward module of ConvLSTM.
        param input: input data with dimensions [batch size, channel, height, width]
        """
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[4])#.cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 512, 64, 32))#.cuda()
    target = Variable(torch.randn(1, 32, 64, 32))#.double().cuda()

    output = convlstm(input)
    output = output[0][0]#.double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)    