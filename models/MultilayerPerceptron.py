import tensorflow as tf
from typing import Tuple, Optional


class MultilayerPerceptron:
    @staticmethod
    def get_simple_mlp(
        flatten: bool,
        input_shape: Tuple,
        activation: str,
        architecture: list,
        output_shape: Optional[int] = 0,
        output_activation: Optional[str] = "",
        reshape_output: Optional[Tuple] = None,
    ):
        input_layer = (
            [tf.keras.layers.Flatten()]
            if flatten
            else [tf.keras.layers.InputLayer(input_shape=input_shape)]
        )
        hidden_layers = [
            tf.keras.layers.Dense(x, activation=activation) for x in architecture
        ]

        if output_shape > 0:
            if len(output_activation):
                output_layer = [
                    tf.keras.layers.Dense(output_shape, activation=activation)
                ]
            else:
                output_layer = [tf.keras.layers.Dense(output_shape)]
            final_architecture = input_layer + hidden_layers + output_layer
        else:
            final_architecture = input_layer + hidden_layers

        if reshape_output is not None:
            reshape_layer = [tf.keras.layers.Reshape(reshape_output)]
            final_architecture += reshape_layer
        mlp = tf.keras.Sequential(final_architecture)
        return mlp
