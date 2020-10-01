import torch
from models.EncoderDecoder.Decoder.Bilinear import BilinearUpsampling
from models.EncoderDecoder.Decoder.BilinearAdditive import BilinearAdditiveUpsampling
from models.EncoderDecoder.Decoder.BilinearSeparableConv import BilinearUpsamplingSeparableConv
from models.EncoderDecoder.Decoder.DecomposedTransposed import DecomposedTransposedConvolution
from models.EncoderDecoder.Decoder.Transposed import TransposedConvolution
from models.EncoderDecoder.Encoder.Encoder import Encoder


class EncoderDecoder(torch.nn.Module):
    def __init__(self, debug=False, decoder_type='transposed'):
        super().__init__()

        assert decoder_type in ['bilinear_additive',
                                'bilinear',
                                'bilinear_seperable',
                                'decomposed_transposed',
                                'transposed', ]

        self.encoder = Encoder(parent_name=self.get_name(),
                               debug=debug)

        if decoder_type == 'bilinear_additive':
            self.decoder = BilinearAdditiveUpsampling(parent_name=self.get_name(),
                                                      debug=debug)
        elif decoder_type == 'bilinear':
            self.decoder = BilinearUpsampling(parent_name=self.get_name(),
                                              debug=debug)
        elif decoder_type == 'bilinear_seperable':
            self.decoder = BilinearUpsamplingSeparableConv(parent_name=self.get_name(),
                                                           debug=debug)
        elif decoder_type == 'decomposed_transposed':
            self.decoder = DecomposedTransposedConvolution(parent_name=self.get_name(),
                                                           debug=debug)
        elif decoder_type == 'transposed':
            self.decoder = TransposedConvolution(parent_name=self.get_name(),
                                                 debug=debug)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x, dims):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    @staticmethod
    def get_static_name():
        raise NotImplementedError
