import torch.nn as nn
from models.EncoderDecoder.Decoder.Decoder import Decoder, UpsampleCell


class UpsamplingCell(UpsampleCell):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         stride=2,
                                         padding=0,
                                         output_padding=1,
                                         kernel_size=2)

    def forward(self, x, dims):
        return self.deconv(x, output_size=dims)


class TransposedConvolution(Decoder):
    def __init__(self,
                 parent_name='',
                 debug=False):
        super().__init__(parent_name=parent_name,
                         debug=debug)

    def create_upsample_layer(self, i, in_channels, out_channels):
        setattr(self, "stage_{}_upsample".format(i),
                UpsamplingCell(in_channels,
                               out_channels)
                )
