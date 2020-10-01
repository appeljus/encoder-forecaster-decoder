import torch.nn as nn
from models.EncoderDecoder.Decoder.Decoder import Decoder, UpsampleCell


class UpsamplingCell(UpsampleCell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 i,
                 kernels_per_layer=3):
        self.resolution_per_layer = [
            363,  # Stage 1
            181,  # Stage 2
            90,  # Stage 3
            45,  # Stage 4
            22,  # Stage 5
            11,  # Stage 6
            5,  # Stage 7
            2  # Stage 8
        ]

        super().__init__()

        self.upsample = nn.Upsample((self.resolution_per_layer[i], self.resolution_per_layer[i]),
                                    mode='bilinear',
                                    align_corners=False)

        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x, dims):
        x = self.upsample(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BilinearUpsamplingSeparableConv(Decoder):
    def __init__(self,
                 parent_name='',
                 debug=False):
        super().__init__(parent_name=parent_name,
                         debug=debug)

    def create_upsample_layer(self, i, in_channels, out_channels):
        setattr(self, "stage_{}_upsample".format(i),
                UpsamplingCell(in_channels, out_channels, i)
                )
