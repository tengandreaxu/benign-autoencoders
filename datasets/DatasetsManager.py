import tensorflow as tf
import numpy as np
from typing import Optional
from utils.activations import sigmoid
from keras.datasets import mnist, fashion_mnist


class DatasetManager:
    """This class handles datasets loading"""

    def __init__(self):
        self.support_datasets = ["fmnist", "cifar-10", "sigmoid"]

    def get_nist_dataset(self, dataset_name: str, noise_factor: float, seed: int):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if dataset_name == "mnist":
            print("Loading MNIST")
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        else:
            print("Loading FMNIST")
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        x_train = np.reshape(x_train, (60000, 28, 28, 1))
        x_test = np.reshape(x_test, (10000, 28, 28, 1))

        # if set, adding noise to the dataset
        x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
        x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

        x_train = tf.clip_by_value(
            x_train_noisy, clip_value_min=0.0, clip_value_max=1.0
        )
        x_test = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_sigmoid_simulated_data(
        sample_size: int,
        n_features: int,
        sigmoid_activation: bool,
        number_neurons: int,
        noise_size: float,
        training_split: Optional[float] = 0.8,
        seed: Optional[int] = 0,
    ):
        """produces a simulated dataset with sigmoid activation function

        Returns
            X_train, X_trest, y_train, y_test from the generated dataset
        """
        np.random.seed(seed)
        x_ = np.random.randn(sample_size, n_features)

        if sigmoid_activation:
            x_ = sigmoid(x_)

        weights = np.random.randn(number_neurons, n_features)

        second_layer_weights = np.random.randn(1, number_neurons)

        noise = noise_size * np.random.randn(sample_size, 1)
        labels = (
            np.exp(x_ @ weights.T) / (1 + np.exp(x_ @ weights.T)) + noise
        ).reshape(x_.shape[0], number_neurons) @ second_layer_weights.T
        training_size = int(training_split * sample_size)
        X_train = x_[:training_size, :]
        y_train = labels[:training_size]
        X_test = x_[training_size:, :]
        y_test = labels[training_size:]
        return X_train, X_test, y_train, y_test
