# Title: ConvLSTM
# Author: Zijie Zhuang (automan000)
# Date: 22-06-2019
# Availability: https://github.com/automan000/Convolutional_LSTM_PyTorch
# Implementation based on Shi et al. (2015).
# Adjusted to fit the existed code of this research by Martijn de Bijl, 06-08-2020.

import torch.nn as nn
import torch
import logging
from torch.autograd import Variable
from models.Forecaster.Forecaster import Forecaster


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

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

    def forward(self, x, current_hidden):
        h, c = current_hidden
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden_dim, image_shape, device):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden_dim, image_shape[0], image_shape[1]))
            self.Wci = self.Wci.to(device)
            self.Wcf = Variable(torch.zeros(1, hidden_dim, image_shape[0], image_shape[1]))
            self.Wcf = self.Wcf.to(device)
            self.Wco = Variable(torch.zeros(1, hidden_dim, image_shape[0], image_shape[1]))
            self.Wco = self.Wco.to(device)
        else:
            assert image_shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert image_shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        h = Variable(torch.zeros(batch_size, hidden_dim, image_shape[0], image_shape[1])).to(device)
        c = Variable(torch.zeros(batch_size, hidden_dim, image_shape[0], image_shape[1])).to(device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dims: Number of channels in input
        hidden_dims: Number of hidden channels
        kernel_sizes: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self,
                 input_dims,
                 hidden_dims,
                 kernel_sizes,
                 num_layers,
                 batch_first=False,
                 return_all_layers=False,
                 debug=False,
                 device=None):
        super().__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.device = device
        self.debug = debug
        kernel_sizes = self._extend_for_multilayer(kernel_sizes, num_layers)
        hidden_dims = self._extend_for_multilayer(hidden_dims, num_layers)
        if not len(kernel_sizes) == len(hidden_dims) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dims
        self.hidden_dim = hidden_dims
        self.kernel_size = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],)
                             )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: Previous hidden state. Should be a list of (h, c) with length num_layers.
        Returns
        -------
        last_state_list, layer_output
        """

        if self.debug:
            self.log("Input shape", input_tensor.size())

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            if self.debug:
                self.log('Permuted input tensor to batch first', input_tensor.size())

        if hidden_state is None:
            hidden_state = self.init_hidden(input_tensor.size()[0], (input_tensor.size()[3], input_tensor.size()[4]), self.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], current_hidden=(h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        if self.debug:
            self.log('Output shape', self.get_size(layer_output_list))
            self.log('Last state list', self.get_size(last_state_list))

        return layer_output_list, last_state_list

    def init_hidden(self, batch_size, image_size_tuple, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, self.hidden_dim[i], image_size_tuple, device))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_static_name(), message, data))

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')


class ConvLSTMWrapper(Forecaster):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 device,
                 return_all_layers=False,
                 debug=False,
                 do_persistence=False,
                 backprop_all_outputs=False):

        super().__init__(device=device,
                         debug=debug,
                         do_persistence=do_persistence,
                         backprop_all_outputs=backprop_all_outputs)

        self.lstm = ConvLSTM(input_dim,
                             hidden_dim,
                             kernel_size,
                             num_layers,
                             batch_first=True,
                             device=device,
                             return_all_layers=return_all_layers)

    def forward_recurrent(self, stacked_encoded_images):
        input_shape_data = self.get_size(stacked_encoded_images)
        x, hidden_state = self.lstm(stacked_encoded_images)

        if self.debug:
            self.log("Shape of data into LSTM", input_shape_data)
            self.log("Shape of output data LSTM", self.get_size(x))
            self.log("Shape of hidden after LSTM", self.get_size(hidden_state))
            self.log("Shape of data from last layer of LSTM", self.get_size(x[-1]))
            self.log("Shape of data into decoder", self.get_size(x[-1][:, -1]))

        if self.backprop_all_outputs:
            result = x[-1]
        else:
            result = x[-1][:, -1]
        return result

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_static_name(), message, data))

    @staticmethod
    def get_static_name():
        return NotImplementedError
