import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, input):
        z = self.encoder(input)
        return z

    def generate(self, z):
        x = self.decoder(z)
        return x
