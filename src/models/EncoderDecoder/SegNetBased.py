from models.EncoderDecoder.EncoderDecoder import EncoderDecoder


class SegNetBased(EncoderDecoder):
    def __init__(self,
                 parent_name='',
                 debug=False,
                 decoder_type='transposed'):

        self.parent_name = parent_name

        super().__init__(debug=debug, decoder_type=decoder_type)

    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """

        # Encode
        x, dims = self.encoder(input_img)

        # Decode
        x = self.decoder(x, dims)

        return x

    def encode(self, x):
        """
        For use in combination with recurrent network.
        :param x: input image
        :return: encoded input image, dimensions of inbetween steps during encoding, and max pool indices of pooling layers during encoding
        """
        x, dims = self.encoder(x)

        return x, dims

    def decode(self, x, dims):
        """
        For use in combination with recurrent network.
        :param x: encoded image
        :param dims: dimensions of inbetween steps during encoding
        :return: reconstructed image
        """
        return self.decoder(x, dims)

    def get_name(self):
        if self.parent_name is not '':
            return '{}_SegNet'.format(self.parent_name)
        else:
            return 'SegNet'

    @staticmethod
    def get_static_name():
        return 'SegNet'
