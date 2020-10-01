import logging

import torch.nn as nn


class PassThrough(nn.Module):
    def __init__(self, parent_name='', debug=False):
        self.debug = debug
        self.parent_name = parent_name
        super().__init__()

    def encode(self, x):
        return x, None

    def decode(self, x, dims):
        if self.debug:
            self.log('Shape of input at decoder', x.size())
        return x

    def forward(self, x):
        return x

    def get_name(self):
        if self.parent_name is not '':
            return '{} - PassThrough'.format(self.parent_name)
        else:
            return 'PassThrough'

    def log(self, name, message):
        logging.info("{} - {} - {}: {}".format(self.parent_name, self.get_name(), name, message))

    @staticmethod
    def get_static_name():
        return 'PassThrough'
