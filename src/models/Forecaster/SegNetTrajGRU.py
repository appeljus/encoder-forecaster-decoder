from models.EncoderDecoder.SegNetBased import SegNetBased
from models.Forecaster.TrajGRU import TrajGRUWrapper


class SegNetTrajGRU(TrajGRUWrapper):
    def __init__(self,
                 input_dim,
                 device,
                 hidden_dims,
                 num_layers,
                 Ls,
                 i2h_kernels,
                 i2h_strides,
                 i2h_pads,
                 i2h_dilates,
                 h2h_kernels,
                 debug=False,
                 do_persistence=False,
                 encoder_decoder_state_dict=None,
                 freeze_encoder_decoder=True,
                 soft_start=False,
                 backprop_all_outputs=False,
                 decoder_type="transposed"):

        super().__init__(input_dim,
                         device,
                         hidden_dims,
                         num_layers,
                         Ls,
                         i2h_kernels,
                         i2h_strides,
                         i2h_pads,
                         i2h_dilates,
                         h2h_kernels,
                         debug,
                         do_persistence,
                         backprop_all_outputs=backprop_all_outputs)

        self.backprop_all_outputs = backprop_all_outputs
        self.encoder_decoder = SegNetBased(parent_name=self.get_static_name(),
                                           debug=debug,
                                           decoder_type=decoder_type)

        self.load_encoder_decoder(encoder_decoder_state_dict, freeze_encoder_decoder, soft_start)

    @staticmethod
    def get_static_name():
        return 'SegNetTrajGRU'
