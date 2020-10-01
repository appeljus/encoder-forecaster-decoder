import torch
import torch.nn as nn
from models.EncoderDecoder.Decoder.Decoder import Decoder, DecoderCell, UpsampleCell


class UpsamplingCell(UpsampleCell):
    def __init__(self,
                 i,
                 ratio=4):
        super().__init__()

        self.ratio = ratio
        self.resolution_per_layer = [
            363,  # Stage 1
            181,  # Stage 2
            90,  # Stage 3
            45,  # Stage 4
            22  # Stage 5
        ]

        self.upsample = nn.Upsample((self.resolution_per_layer[i], self.resolution_per_layer[i]),
                                    mode='bilinear',
                                    align_corners=True)

    def forward(self, x, dims):
        in_channels = x.size()[1]
        out_channels = int(in_channels / self.ratio)
        x = self.upsample(x)
        output_list = []
        for i in range(out_channels):
            output_list.append(torch.mean(x[:, i*self.ratio:(i+1)*self.ratio, ...], dim=1))

        x = torch.stack(output_list, 1)
        return x


class BilinearAdditiveUpsampling(Decoder):
    def __init__(self,
                 parent_name='',
                 debug=False):
        out_channel_list = [1, 2, 8, 32, 128, 512]
        super().__init__(parent_name,
                         debug,
                         out_channel_list=out_channel_list)

    def create_upsample_layer(self, i, in_channels, out_channels):
        if i == 0:
            ratio = 2
        else:
            ratio = 4
        setattr(self, "stage_{}_upsample".format(i),
                UpsamplingCell(i, ratio)
                )

    def create_layer(self, i, j, in_channels, out_channels):
        setattr(self, "stage_{}_layer_{}".format(i, j),
                DecoderCell(in_channels,
                            in_channels,
                            stride=1,
                            padding=1,
                            kernel_size=3)
                )
