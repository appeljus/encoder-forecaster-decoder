import torch.nn as nn
from models.EncoderDecoder.Decoder.Decoder import Decoder, UpsampleCell


class UpsamplingCell(UpsampleCell):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.vertical_deconv = nn.ConvTranspose2d(in_channels,
                                                  out_channels,
                                                  stride=(2, 1),
                                                  kernel_size=(2, 1),
                                                  padding=(0, 0),
                                                  output_padding=(1, 0)
                                                  )

        self.horizontal_deconv = nn.ConvTranspose2d(in_channels,
                                                    out_channels,
                                                    stride=(1, 2),
                                                    kernel_size=(1, 2),
                                                    padding=(0, 0),
                                                    output_padding=(0, 1))

    def forward(self, x, dims):
        horizontal = self.horizontal_deconv(x, output_size=[x.size()[2], dims[3]])
        vertical = self.vertical_deconv(horizontal, output_size=[dims[2], horizontal.size()[3]])
        return vertical


class DecomposedTransposedConvolution(Decoder):
    def __init__(self,
                 parent_name='',
                 debug=False):
        super().__init__(parent_name=parent_name,
                         debug=debug)

    def create_upsample_layer(self, i, in_channels, out_channels):
        setattr(self, "stage_{}_upsample".format(i),
                UpsamplingCell(in_channels, out_channels)
                )
