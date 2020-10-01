import logging
import torch


class UpsampleCell(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dims):
        return NotImplementedError


class DecoderCell(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding=1,
                 stride=1,
                 kernel_size=3):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.nn.functional.relu(self.batch_norm(self.conv(x)))


class Decoder(torch.nn.Module):
    def __init__(self,
                 parent_name='',
                 debug=False,
                 out_channel_list=None):
        super().__init__()

        self.stages = 5
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
        if out_channel_list is None:
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
        else:
            self.out_channel_list = out_channel_list

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

        for i in range(self.stages - 1, -1, -1):
            self.create_upsample_layer(i, self.out_channel_list[i], self.out_channel_list[i])
            for j in range(self.conv_per_layer[i] - 1, -1, -1):
                if j == 0:
                    self.create_layer(i, j, self.out_channel_list[i], self.in_channel_list[i])
                else:
                    self.create_layer(i, j, self.out_channel_list[i], self.out_channel_list[i])

    def create_upsample_layer(self, i, in_channels, out_channels):
        return NotImplementedError

    def create_layer(self, i, j, in_channels, out_channels):
        setattr(self, "stage_{}_layer_{}".format(i, j),
                DecoderCell(in_channels,
                            out_channels,
                            stride=1,
                            padding=1,
                            kernel_size=3)
                )

    def forward(self, x, dims):
        for i in range(self.stages - 1, -1, -1):
            upsample_layer = getattr(self, "stage_{}_upsample".format(i))
            x = upsample_layer(x, dims[i])

            if self.debug:
                self.log("Size after upsamling in stage {}".format(i), x.size())

            for j in range(self.conv_per_layer[i] - 1, -1, -1):
                layer = getattr(self, "stage_{}_layer_{}".format(i, j))
                x = layer(x)

                if self.debug:
                    self.log("Size after stage {}, layer {}".format(i, j), x.size())

        return x

    def log(self, label, data):
        logging.info("{} - {} - {}: {}".format(self.parent_name, self.get_name(), label, data))

    @staticmethod
    def get_name():
        return "Decoder"
