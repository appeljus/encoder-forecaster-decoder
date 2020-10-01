# Title: Forecaster
# Author: Zhizhing Huang (Hzzone)
# Date: 22-06-2019
# Availability: https://github.com/Hzzone/Precipitation-Nowcasting
# Adjusted to fit the existed code of this research by Martijn de Bijl, 06-08-2020.

import logging

from torch import nn
import torch
from evaluation.models.convRNN.utils import make_layers


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, debug=False, seq_len=3):
        super().__init__()
        self.seq_len = seq_len
        self.debug = debug
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn, stage_index):
        input, state_stage = rnn(input, state, seq_len=self.seq_len)

        if self.debug:
            self.log('output after rnn{0}'.format(stage_index), self.get_size(input))

        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)

        if self.debug:
            self.log('output after stage{0}'.format(stage_index), self.get_size(input))
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        if self.debug:
            self.log('size of input hidden', self.get_size(hidden_states))
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'), 3)
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)), i)
        return input

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_name(), message, data))

    @staticmethod
    def get_name():
        return "Forecaster"

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')
