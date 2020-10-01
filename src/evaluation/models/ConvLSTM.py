from evaluation.models.convRNN.forecaster import Forecaster
from evaluation.models.convRNN.encoder import Encoder
from evaluation.models.convRNN.model import EF
from evaluation.models.convRNN.net_params import get_conv_lstm_encoder_params, get_conv_lstm_forecaster_params
from models.EncoderDecoder.PassThrough import PassThrough
from models.Recurrent.Recurrent import Recurrent


class ConvLSTM(Recurrent):
    def __init__(self,
                 device,
                 do_persistence,
                 backprop_all_outputs=False,
                 debug=False,
                 seq_len=3,
                 batch_size=1,
                 forecast_horizon=1,
                 model_params=None):
        super().__init__(device, debug, do_persistence, backprop_all_outputs)

        self.backprop_all_outputs = backprop_all_outputs
        self.debug = debug
        self.do_persistence = do_persistence

        encoder_params = get_conv_lstm_encoder_params(batch_size, device)
        forecaster_params = get_conv_lstm_forecaster_params(batch_size, device)
        encoder = Encoder(encoder_params[0], encoder_params[1], debug=debug).to(device)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1], debug=debug, seq_len=seq_len).to(device)

        self.encoder_forecaster = EF(encoder, forecaster).to(device)

        self.encoder_decoder = PassThrough()

    def forward_recurrent(self, stacked_encoded_images):
        # B, S, C, H, W -> S, B, C, H, W
        stacked_encoded_images = stacked_encoded_images.permute(1, 0, 2, 3, 4)
        result = self.encoder_forecaster(stacked_encoded_images)
        return result[-1]

    @staticmethod
    def get_static_name():
        return "ConvLSTM"
