from keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
)
from keras.models import Model, Sequential


class KerasMNISTAE(Model):
    """
    source code:
    https://blog.keras.io/building-autoencoders-in-keras.html
    """

    def __init__(self):
        super().__init__()
        self.encoder = KerasMNISTAE.get_encoder()
        self.decoder = KerasMNISTAE.get_decoder()

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    @staticmethod
    def get_encoder() -> Model:
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(28, 28, 1)))
        # 28x28x16
        encoder.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        encoder.add(BatchNormalization())
        # 14x14x16
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(BatchNormalization())
        # 14x14x8
        encoder.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        encoder.add(BatchNormalization())
        # 7x7x8
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(BatchNormalization())
        # 7x7x8
        encoder.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        encoder.add(BatchNormalization())
        # 4x4x8
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(BatchNormalization())
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        return encoder

    @staticmethod
    def get_decoder() -> Model:

        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(4, 4, 8)))
        decoder.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(BatchNormalization())
        decoder.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(BatchNormalization())
        decoder.add(Conv2D(16, (3, 3), activation="relu"))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
        decoder.add(BatchNormalization())
        return decoder
