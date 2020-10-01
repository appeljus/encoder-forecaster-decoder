import torch.nn as nn
import logging

F = nn.functional


class Encoder(nn.Module):
    def __init__(self, parent_name='', debug=False):
        super().__init__()

        self.encoding_stages = 5
        self.parent_name = parent_name
        self.debug = debug

        self.in_channel_list = [
            1,  # Stage 1
            64,  # Stage 2
            128,  # Stage 3
            256,  # Stage 4
            512,  # Stage 5
            512,  # Stage 6
            1024,  # Stage 7
            1024  # Stage 8
        ]
        self.out_channel_list = [
            64,  # Stage 1
            128,  # Stage 2
            256,  # Stage 3
            512,  # Stage 4
            512,  # Stage 5
            1024,  # Stage 6
            1024,  # Stage 7
            1024  # Stage 8
        ]
        self.conv_per_layer = [
            2,  # Stage 1
            2,  # Stage 2
            3,  # Stage 3
            3,  # Stage 4
            3,  # Stage 5
            3,  # Stage 6
            3,  # Stage 7
            3  # Stage 8
        ]
        for i in range(self.encoding_stages):
            for j in range(self.conv_per_layer[i]):
                if i == 7 and j == 2:
                    self.create_layer(i, j, self.out_channel_list[i], self.out_channel_list[i])
                elif j == 0:
                    self.create_layer(i, j, self.in_channel_list[i], self.out_channel_list[i])
                else:
                    self.create_layer(i, j, self.out_channel_list[i], self.out_channel_list[i])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def create_layer(self, i, j, in_channels, out_channels, stride=1):
        setattr(self, "encoder_conv_{}{}".format(i, j),
                nn.Sequential(*[
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              stride=stride,
                              ),
                    nn.BatchNorm2d(out_channels)
                ])
                )

    def forward(self, x):
        input_size = x.size()
        dims = [input_size]
        for i in range(self.encoding_stages):
            for j in range(self.conv_per_layer[i]):
                x = F.relu(getattr(self, "encoder_conv_{}{}".format(i, j))(x))

                if self.debug:
                    self.log('Size after stage {}, layer {}'.format(i, j), x.size())

            x, current_indices = self.pool(x)
            dims.append(x.size())

            if self.debug:
                self.log('Size after pooling in stage {}'.format(i), x.size())

        return x, dims

    def log(self, name, message):
        logging.info("{} - {} - {}: {}".format(self.parent_name, self.get_name(), name, message))

    def get_name(self):
        return "Encoder"
