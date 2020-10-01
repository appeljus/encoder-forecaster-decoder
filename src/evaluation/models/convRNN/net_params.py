# Title: Network parameters
# Author: Zhizhing Huang (Hzzone)
# Date: 22-06-2019
# Availability: https://github.com/Hzzone/Precipitation-Nowcasting
# Adjusted to fit the existed code of this research by Martijn de Bijl, 06-08-2020.

from collections import OrderedDict
from evaluation.models.convRNN.TrajGRUModel import TrajGRU
from evaluation.models.convRNN.ConvLSTMModel import ConvLSTM
import torch.nn.functional as F


# build model
def get_traj_gru_encoder_params(batch_size, device):
    return [
        [
            # in_channel, out_channels, kernel_size, stride, padding
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],
        [
            TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 72, 72), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device),
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device),
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 12, 12), zoneout=0.0, L=9,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device)
        ]
    ]


def get_traj_gru_forecaster_params(batch_size, device):
    return [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1, 0]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1, 0]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 7, 5, 1, 3],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 12, 12), zoneout=0.0, L=9,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device),
            TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device),
            TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 72, 72), zoneout=0.0, L=13,
                    i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                    h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                    act_type=F.relu, device=device)
        ]
    ]


def get_conv2d_params():
    return OrderedDict({
        'conv1_relu_1': [5, 64, 7, 5, 1],
        'conv2_relu_1': [64, 192, 5, 3, 1],
        'conv3_relu_1': [192, 192, 3, 2, 1],
        'deconv1_relu_1': [192, 192, 4, 2, 1, 0],
        'deconv2_relu_1': [192, 64, 5, 3, 1, 0],
        'deconv3_relu_1': [64, 64, 7, 5, 1, 0],
        'conv3_relu_2': [64, 20, 3, 1, 1],
        'conv3_3': [20, 20, 1, 1, 0]
    })


def get_conv_lstm_encoder_params(batch_size, device):
    return [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],
        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 72, 72),
                     kernel_size=3, stride=1, padding=1, device=device),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24),
                     kernel_size=3, stride=1, padding=1, device=device),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 12, 12),
                     kernel_size=3, stride=1, padding=1, device=device),
        ]
    ]


def get_conv_lstm_forecaster_params(batch_size, device):
    return [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1, 0]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1, 0]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 7, 5, 1, 3],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 12, 12),
                     kernel_size=3, stride=1, padding=1, device=device),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24),
                     kernel_size=3, stride=1, padding=1, device=device),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 72, 72),
                     kernel_size=3, stride=1, padding=1, device=device),
        ]
    ]
