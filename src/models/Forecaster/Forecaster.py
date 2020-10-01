import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self,
                 device,
                 debug=False,
                 do_persistence=False,
                 backprop_all_outputs=False):

        super().__init__()

        self.backprop_all_outputs = backprop_all_outputs
        self.do_persistence = do_persistence
        self.device = device
        self.debug = debug

        if self.do_persistence:
            self.forward_func = self.forward_with_persistence
        else:
            self.forward_func = self.forward_without_persistence

    def load_encoder_decoder(self, encoder_decoder_state_dict, freeze_encoder_decoder, soft_start):
        if soft_start:
            self.encoder_decoder.load_state_dict(encoder_decoder_state_dict)
        if freeze_encoder_decoder:
            for param in self.encoder_decoder.parameters():
                param.requires_grad = False

    def forward(self, **kwargs):
        for key, value in kwargs.items():
            x, dims = self.encoder_decoder.encode(value)
            setattr(self, key, x)
        stacked_encoded_images = torch.stack([getattr(self, x) for x in kwargs.keys()], dim=1)

        # Stacked encoded images: B, S, C, W, H
        return self.forward_func(stacked_encoded_images, dims)

    def forward_recurrent(self, stacked_encoded_images):
        return NotImplementedError

    def forward_with_persistence(self, stacked_encoded_images, dims):
        x = self.forward_recurrent(stacked_encoded_images)

        if self.backprop_all_outputs:
            seq_len = stacked_encoded_images.shape[1]
            result = torch.zeros([1, seq_len, 1, 363, 363]).to(self.device)
            for i in range(seq_len):
                result[:, i] = self.encoder_decoder.decode(x[:, i], dims)
        else:
            result = self.encoder_decoder.decode(x, dims)

        persistence = self.encoder_decoder.decode(stacked_encoded_images[:, -1], dims)

        return result, persistence

    def forward_without_persistence(self, stacked_encoded_images, dims):
        x = self.forward_recurrent(stacked_encoded_images)

        x = self.encoder_decoder.decode(x, dims)

        return x

    @staticmethod
    def get_static_name():
        return NotImplementedError

    def get_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        elif isinstance(data, tuple):
            return [self.get_size(x) for x in data]
        elif isinstance(data, list):
            return [self.get_size(x) for x in data]
        else:
            return TypeError('Data should be one of [Tensor, tuple, list]')
