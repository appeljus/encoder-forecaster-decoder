# Title: Forecaster
# Author: Zhizhing Huang (Hzzone)
# Date: 22-06-2019
# Availability: https://github.com/Hzzone/Precipitation-Nowcasting
# Implementation based on Shi et al. (2017).
# Adjusted to fit the existed code of this research by Martijn de Bijl, 06-08-2020.

import logging
import torch
from torch import nn
import torch.nn.functional as F
from models.Forecaster.Forecaster import Forecaster


class TrajGRUCell(nn.Module):
    def __init__(self,
                 index,
                 debug,
                 input_dim,
                 hidden_dim,
                 device,
                 zoneout=0.0,
                 L=5,
                 i2h_kernel=(3, 3),
                 i2h_stride=(1, 1),
                 i2h_pad=(1, 1),
                 i2h_dilate=(1, 1),
                 h2h_kernel=(5, 5),
                 activation=torch.tanh):
        """

        :param input_dim:
        :param hidden_dim:
        :param zoneout:
        :param L: is the total number of allowed links
        :param i2h_kernel:
        :param i2h_stride:
        :param i2h_pad:
        :param i2h_dilate:
        :param h2h_kernel:
        :param activation:
        """
        super().__init__()

        self.debug = debug
        self.index = index
        self.device = device
        assert (h2h_kernel % 2 == 1), "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self.activation = activation
        self.L = L
        self.zoneout = zoneout
        self.hidden_dim = hidden_dim

        self.i2h_kernel = i2h_kernel
        self.i2h_stride = i2h_stride
        self.i2h_pad = i2h_pad
        self.i2h_dilate = i2h_dilate

        # Correspond wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_dim,
                             out_channels=hidden_dim * 3,
                             kernel_size=i2h_kernel,
                             stride=i2h_stride,
                             padding=i2h_pad,
                             dilation=i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_dim,
                                   out_channels=32,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2,
                                   dilation=1)

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=hidden_dim,
                                   out_channels=32,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2,
                                   dilation=1)

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=32,
                                    out_channels=self.L * 2,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

        # Correspond hh, hz, hr for 1 * 1 convolutional kernel
        self.ret = nn.Conv2d(in_channels=hidden_dim * self.L,
                             out_channels=hidden_dim * 3,
                             kernel_size=1,
                             stride=1)

    # inputs: B*C*H*W
    def _flow_generator(self, inputs, hidden_states):
        i2f_conv1 = self.i2f_conv1(inputs)
        h2f_conv1 = self.h2f_conv1(hidden_states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self.activation(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)

        return flows

    # inputs with states empty at different times
    # inputs: S*B*C*H*W
    def forward(self, inputs, hidden_state):
        seq_len, batch, channel, height, width = inputs.size()
        i2h = self.i2h(torch.reshape(inputs, (-1, channel, height, width)))
        i2h = torch.reshape(i2h, (seq_len, batch, i2h.size(1), i2h.size(2), i2h.size(3)))
        i2h_slice = torch.split(i2h, self.hidden_dim, dim=2)

        previous_hidden = hidden_state
        outputs = []
        for i in range(seq_len):
            flows = self._flow_generator(inputs[i], previous_hidden)
            wrapped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                wrapped_data.append(self.wrap(previous_hidden, -flow, self.device))
            wrapped_data = torch.cat(wrapped_data, dim=1)
            h2h = self.ret(wrapped_data)
            h2h_slice = torch.split(h2h, self.hidden_dim, dim=1)
            reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
            update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
            new_mem = self.activation(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            next_hidden = update_gate * previous_hidden + (1 - update_gate) * new_mem

            if self.debug:
                self.log("reset gate at t={}".format(i), self.get_size(reset_gate))
                self.log("update gate at t={}".format(i), self.get_size(update_gate))
                self.log("next hidden at t={}".format(i), self.get_size(next_hidden))

            if self.zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(previous_hidden), p=self.zoneout)
                next_hidden = torch.where(mask, next_hidden, previous_hidden)

            outputs.append(next_hidden)
            previous_hidden = next_hidden

        return torch.stack(outputs), next_hidden

    def wrap(self, input, flow, device):
        B, C, H, W = input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid, align_corners=False)

        return output

    def init_hidden(self, data_dims, device):
        i2h_dilate_ksize = 1 + (self.i2h_kernel - 1) * self.i2h_dilate
        _, batch_dim, _, height, width = data_dims
        hidden_state_size = (height + 2 * self.i2h_pad - i2h_dilate_ksize) // self.i2h_stride + 1

        return torch.zeros((batch_dim, self.hidden_dim, hidden_state_size, hidden_state_size), dtype=torch.float).to(device)

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_name(), message, data))

    def get_name(self):
        return "TrajGRUCell_{}".format(self.index)

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')


class TrajGRU(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 num_layers,
                 Ls,
                 i2h_kernels,
                 i2h_strides,
                 i2h_pads,
                 i2h_dilates,
                 h2h_kernels,
                 log_func,
                 return_all_layers=False,
                 batch_first=True,
                 debug=False,
                 device=None
                 ):
        """

        :param input_dim:
        :param hidden_dims:
        :param kernel_sizes:
        :param num_layers:
        :param batch_first: True if dimension 0 is batch, else False
        :param debug:
        :param device:
        """
        super().__init__()

        self.log = log_func
        self.return_all_layers = return_all_layers
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.debug = debug
        self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else hidden_dims[i - 1]

            cell_list.append(TrajGRUCell(i,
                                         debug,
                                         cur_input_dim,
                                         hidden_dims[i],
                                         device,
                                         zoneout=0.0,
                                         L=Ls[i],
                                         i2h_kernel=i2h_kernels[i],
                                         i2h_stride=i2h_strides[i],
                                         i2h_pad=i2h_pads[i],
                                         i2h_dilate=i2h_dilates[i],
                                         h2h_kernel=h2h_kernels[i],
                                         activation=torch.tanh)
                             )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, stacked_encoded_images, hidden_state=None):
        if self.batch_first:
            # (batch, time, channel, height, width) -> (time, batch, channel, height, width)
            stacked_encoded_images = stacked_encoded_images.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self.init_hidden(stacked_encoded_images.size(), self.device)

        layer_output_list = []
        last_hidden_state_list = []

        cur_layer_input = stacked_encoded_images

        for layer_idx in range(self.num_layers):
            layer_hidden_state = hidden_state[layer_idx]

            # Cell params: inputs, hidden_state
            # Cell returns: outputs, last_hidden
            layer_outputs, last_hidden = self.cell_list[layer_idx](cur_layer_input, layer_hidden_state)
            cur_layer_input = layer_outputs

            layer_output_list.append(layer_outputs)
            last_hidden_state_list.append(last_hidden)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]

        if self.debug:
            self.log('Output shape', self.get_size(layer_output_list))
            self.log('Last state list', self.get_size(last_hidden_state_list))

        return layer_output_list, last_hidden_state_list

    def init_hidden(self, data_dims, device):
        hidden_state = []
        for layer_idx in range(self.num_layers):
            hidden_state.append(self.cell_list[layer_idx].init_hidden(data_dims, device))
        return hidden_state

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')


class TrajGRUWrapper(Forecaster):
    def __init__(self,
                 input_dim,
                 device,
                 hidden_dims,
                 num_layers,
                 Ls,
                 i2h_kernels,
                 i2h_strides,
                 i2h_pads,
                 i2h_dilates,
                 h2h_kernels,
                 debug=False,
                 do_persistence=False,
                 backprop_all_outputs=False):

        super().__init__(device,
                         debug,
                         do_persistence,
                         backprop_all_outputs=backprop_all_outputs)

        self.gru = TrajGRU(input_dim,
                           hidden_dims,
                           num_layers,
                           Ls,
                           i2h_kernels,
                           i2h_strides,
                           i2h_pads,
                           i2h_dilates,
                           h2h_kernels,
                           log_func=self.log,
                           batch_first=True,
                           debug=debug,
                           device=device
                           )

    def forward_recurrent(self, stacked_encoded_images):
        input_shape_data = self.get_size(stacked_encoded_images)

        if self.debug:
            self.log("Shape of data into GRU", input_shape_data)
        x, hidden_state = self.gru(stacked_encoded_images)
        if self.debug:
            self.log("Shape of output data GRU", self.get_size(x))
            self.log("Shape of hidden after GRU", self.get_size(hidden_state))
            self.log("Shape of data into decoder", self.get_size(x[-1]))

        if self.backprop_all_outputs:
            result = x.permute(1, 0, 2, 3, 4)
        else:
            result = x[-1]
        return result

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_static_name(), message, data))

    @staticmethod
    def get_static_name():
        return "TrajGRU"
