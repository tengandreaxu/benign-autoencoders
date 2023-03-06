from builtins import breakpoint
import os
import tensorflow as tf
from copy import deepcopy
import numpy as np
from keras.models import Model, save_model
from keras.optimizers import Optimizer
from keras.losses import (
    Loss,
    MeanSquaredError,
    SparseCategoricalCrossentropy,
    BinaryCrossentropy,
)
from keras.metrics import Metric
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)


class BenignCompression(Model):
    def __init__(
        self,
        encoder: Model,
        decoder: Model,
        discriminator: Model,
        optimizer: Optimizer,
        loss: Loss,
        metric: Metric,
        reconstruction_loss: Loss,
        decompression_weight: float,
        discrimination_weight: float,
        save_dir: Optional[str] = "",
        batch_size: Optional[int] = 32,
        debug: Optional[bool] = False,
        skip_nn: Optional[bool] = False,
    ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.compression_errors = []
        self.entropies = []
        self.train_entropies = []
        self.train_compression_errors = []
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.decompression_weight = decompression_weight
        self.discrimination_weight = discrimination_weight
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.debug = debug
        self.reconstruction_loss = reconstruction_loss
        self.skip_nn = skip_nn
        self.logger = logging.getLogger("BAE")

    def call(self, inputs):

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        # mse = BinaryCrossentropy()  # MeanSquaredError()
        self.add_loss(self.reconstruction_loss(inputs, decoded) * self.current_weight)

        discriminated = self.discriminator(decoded)

        return discriminated

    def train_part_of_big_nn(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epoch,
        training_type: int,
        is_odd_epoch,
        verbose: Optional[int] = 1,
    ):

        weights = self.get_weights()
        training_info_dict = {
            0: {
                "encoder": [True, False],
                "decoder": [False, True],
                "discriminator": [True, False],
            },
            1: {
                "encoder": [True, True],
                "decoder": [False, True],
                "discriminator": [True, False],
            },
            2: {
                "encoder": [True, True],
                "decoder": [True, True],
                "discriminator": [True, False],
            },
            3: {
                "encoder": [False, True],
                "decoder": [False, True],
                "discriminator": [True, False],
            },
        }

        self.encoder.trainable = training_info_dict[training_type]["encoder"][
            is_odd_epoch
        ]
        self.decoder.trainable = training_info_dict[training_type]["decoder"][
            is_odd_epoch
        ]
        self.discriminator.trainable = training_info_dict[training_type][
            "discriminator"
        ][is_odd_epoch]

        print(f"Weight:\t{self.current_weight:.3f} Is Decoder Epoch:\t{is_odd_epoch}")
        self.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metric,
            loss_weights=(1 - self.current_weight),
        )

        self.set_weights(weights)
        fit_results = self.fit(
            x=x_train,
            y=y_train,
            epochs=epoch + 1,
            initial_epoch=epoch,
            shuffle=True,
            validation_data=(x_test, y_test),
            verbose=verbose,
            batch_size=self.batch_size,
        )

        return fit_results.history

    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs: int,
        training_type: int,
        verbose: Optional[int] = 1,
    ):

        loss = []
        accuracy = []
        val_loss = []
        val_accuracy = []

        mse1 = MeanSquaredError()
        # TODO could be False
        sparse_accuracy = SparseCategoricalCrossentropy(from_logits=True)

        for epoch in np.arange(0, epochs, dtype=int):
            is_decoder_epoch = epoch % 2 == 1

            if self.skip_nn and not is_decoder_epoch:
                continue

            self.current_weight = (
                self.decompression_weight
                if is_decoder_epoch
                else self.discrimination_weight
            )

            fit_results = self.train_part_of_big_nn(
                x_train,
                y_train,
                x_test,
                y_test,
                epoch,
                training_type,
                is_decoder_epoch,
                verbose,
            )

            test_mse = (
                self.current_weight
                * mse1(x_test, self.decoder(self.encoder(x_test))).numpy()
            )

            self.compression_errors.append(test_mse)

            # *************
            # Manual Train errors
            # *************
            if self.debug:
                test_entropy = (1 - self.current_weight) * sparse_accuracy(
                    y_test, self(x_test)
                ).numpy()
                self.entropies.append(test_entropy)
                train_entropy = (1 - self.current_weight) * sparse_accuracy(
                    y_train, self(x_train)
                )

                train_mse = (
                    self.current_weight
                    * mse1(x_train, self.decoder(self.encoder(x_train))).numpy()
                )
                self.logger.info(
                    f"Train Entropy: {train_entropy}\tTrain MSE: {train_mse}"
                )
                self.train_compression_errors.append(train_mse)
                self.train_entropies.append(train_entropy)
                self.logger.info(f"Entropy: {test_entropy}\tMSE: {test_mse}")
            # When training type 3
            # Train MSE in odd epochs
            # Train Entropy in even epochs
            loss.append(deepcopy(fit_results["loss"][0]))

            # When training type 3
            # Test MSE in odd epochs
            # Test Entropy in even epochs
            val_loss.append(deepcopy(fit_results["val_loss"][0]))
            if len(self.metric) > 0:

                score = "accuracy"
                val_score = "val_accuracy"
                if self.metric[0] == "mse":
                    score = "mse"
                    val_score = "val_mse"
                accuracy.append(deepcopy(fit_results[score][0]))
                val_accuracy.append(deepcopy(fit_results[val_score][0]))

        return {
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    def save_autoencoder(self):
        with open(os.path.join(self.save_dir, "AE_architecture.txt"), "w") as f:
            self.encoder.summary(print_fn=lambda x: f.write(x + "\n"))
            self.decoder.summary(print_fn=lambda x: f.write(x + "\n"))
