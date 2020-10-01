from models.EncoderDecoder.SegNetBased import SegNetBased
from models.Forecaster.ConvLSTM import ConvLSTMWrapper


class SegNetConvLSTM(ConvLSTMWrapper):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 device,
                 return_all_layers=False,
                 debug=False,
                 soft_start=False,
                 encoder_decoder_state_dict=None,
                 freeze_encoder_decoder=False,
                 do_persistence=False,
                 backprop_all_outputs=False,
                 decoder_type="transposed"):

        super().__init__(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         kernel_size=kernel_size,
                         num_layers=num_layers,
                         device=device,
                         return_all_layers=return_all_layers,
                         debug=debug,
                         do_persistence=do_persistence,
                         backprop_all_outputs=backprop_all_outputs)

        self.backprop_all_outputs = backprop_all_outputs
        self.encoder_decoder = SegNetBased(parent_name=self.get_static_name(),
                                           debug=debug,
                                           decoder_type=decoder_type)

        self.load_encoder_decoder(encoder_decoder_state_dict, freeze_encoder_decoder, soft_start)

    @staticmethod
    def get_static_name():
        return 'SegNetConvLSTM'
