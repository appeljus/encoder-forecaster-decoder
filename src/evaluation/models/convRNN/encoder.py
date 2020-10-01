# Title: Encoder
# Author: Zhizhing Huang (Hzzone)
# Date: 22-06-2019
# Availability: https://github.com/Hzzone/Precipitation-Nowcasting
# Adjusted to fit the existed code of this research by Martijn de Bijl, 06-08-2020.

from torch import nn
import torch
from evaluation.models.convRNN.utils import make_layers
import logging


class Encoder(nn.Module):
    def __init__(self, subnets, rnns, debug=False):
        super().__init__()
        self.debug = debug
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn, stages_index):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)

        if self.debug:
            self.log('size after stage{}'.format(stages_index), self.get_size(input))

        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, None)

        if self.debug:
            self.log('data size after rnn{}'.format(stages_index), self.get_size(outputs_stage))
            self.log('hidden size after rnn{}'.format(stages_index), self.get_size(state_stage))

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []

        if self.debug:
            self.log('input data', self.get_size(input))

        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)), i)
            hidden_states.append(state_stage)

        if self.debug:
            self.log('output of encoder', self.get_size(tuple(hidden_states)))

        return tuple(hidden_states)

    def log(self, message, data):
        logging.info("{} - {}: {}".format(self.get_name(), message, data))

    @staticmethod
    def get_name():
        return "Encoder"

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')
