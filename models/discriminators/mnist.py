from keras.models import Model, Sequential
from keras.layers import Flatten, Dense


class Discriminator(Model):
    def __init__(self):

        super().__init__()

        self.discriminator = Sequential(
            [Flatten(input_shape=(28, 28, 1))]
            + [Dense(x, activation="relu") for x in [64, 32, 16]]
            + [Dense(10)]
        )

    def call(self, inputs):

        discriminated = self.discriminator(inputs)
        return discriminated
