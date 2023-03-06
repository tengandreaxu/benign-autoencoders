from keras.models import Model


class AutoEncoder(Model):
    """Generic AutoEncoder"""

    def __init__(self, encoder: Model, decoder: Model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
