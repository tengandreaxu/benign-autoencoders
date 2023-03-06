"""
This MNIST ad-hoc autoencoder was found in:
https://www.kaggle.com/code/milan400/cifar10-autoencoder
All the credits goes to the author.
"""
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    InputLayer,
    Dense,
    Flatten,
    Reshape,
    BatchNormalization
)
from keras.models import Model, Sequential


class KaggleMNISTAE(Model):
    def __init__(self):
        super().__init__()
        self.encoder = KaggleMNISTAE.get_encoder()
        self.decoder = KaggleMNISTAE.get_decoder()

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return decoded

    @staticmethod
    def get_encoder() -> Model:

        encoder = Sequential()

        # encoder
        encoder.add(
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),
            )
        )
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling2D((2, 2), padding="same", name="encoder"))
        encoder.add(BatchNormalization())
        encoder.add(Flatten())
        encoder.add(Dense(128))
        encoder.add(BatchNormalization())
        encoder.add(Reshape(target_shape=(4, 4, 8)))
        return encoder

    @staticmethod
    def get_decoder() -> Model:

        decoder = Sequential()

        decoder.add(
            Conv2D(8, (3, 3), activation="relu", padding="same", input_shape=(4, 4, 8))
        )
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(BatchNormalization())
        decoder.add(Conv2D(16, (2, 2), activation="relu", padding="valid"))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(BatchNormalization())
        decoder.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling2D((2, 2)))  
        decoder.add(BatchNormalization())
        decoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
        return decoder
