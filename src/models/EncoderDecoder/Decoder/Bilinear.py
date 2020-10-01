import torch.nn as nn
from models.EncoderDecoder.Decoder.Decoder import Decoder, UpsampleCell


class UpsamplingCell(UpsampleCell):
    def __init__(self, output_resolution):
        super().__init__()

        self.upsample = nn.Upsample(output_resolution,
                                    mode='bilinear',
                                    align_corners=False)

    def forward(self, x, dims):
        return self.upsample(x)


class BilinearUpsampling(Decoder):
    def __init__(self,
                 parent_name='',
                 debug=False):
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
        super().__init__(parent_name=parent_name,
                         debug=debug)

    def create_upsample_layer(self, i, in_channels, out_channels):
        setattr(self, "stage_{}_upsample".format(i),
                UpsamplingCell((self.resolution_per_layer[i], self.resolution_per_layer[i]))
                )
