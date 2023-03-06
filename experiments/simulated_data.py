import argparse
import os
from typing import Optional
import numpy as np
from models.AutoEncoder import AutoEncoder
import pandas as pd
from datasets.DatasetsManager import DatasetManager
from models.MultilayerPerceptron import MultilayerPerceptron
from models.BenignCompression import BenignCompression
from utils.helper_functions import save_pickle
from utils.saving_results import FileNames
from keras.optimizers import Adam
from collections import namedtuple
from keras.losses import MeanSquaredError
from keras.layers import Dense, InputLayer
from keras.models import Sequential
import logging

logging.basicConfig(level=logging.INFO)


class SimulationParameters:
    n_features = [10, 50, 100]
    neurons = [1, 5, 10]
    dws = [0.9]
    nws = [0.1]
    sample_sizes = {10: 10 * 100, 50: 50 * 100, 100: 100 * 100}


def get_nus(n_features):
    if n_features == 10:
        nus = [1, 5, 10, 25]  # [1, 3, 5,7, 9, 10]
    elif n_features == 50:
        nus = [1, 5, 10, 25]  # [1, 5, 10, 25, 35,45]
    elif n_features == 100:
        nus = [1, 5, 10, 25]  # [1,5,10, 25, 50, 75]
    return nus


def get_autoencoder(
    input_shape: int, latent_dimension: int, split: Optional[bool] = False
):
    """returns a shallo autoencoder where the latent space dimension = :latent_dimension
    and input and output shape is equal to the dataset orginal dimension"""
    encoder = Sequential()
    decoder = Sequential()
    # [d, d, d] reaches 0 mse
    encoder.add(InputLayer(input_shape=input_shape))
    # Bottleneck
    encoder.add(Dense(latent_dimension, activation="linear"))
    decoder.add(InputLayer(input_shape=latent_dimension))

    # OutputLayer
    decoder.add(Dense(input_shape, activation="linear"))
    if not split:
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder)
        return autoencoder
    else:
        return encoder, decoder


def run_classic_loss(
    architecture: list,
    input_shape: int,
    epochs: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pickles_dir: str,
    csvs_dir: str,
    seed: int,
):
    csv_file = os.path.join(csvs_dir, f"seed_{seed}_classic_loss.csv")
    pickle_file = os.path.join(pickles_dir, f"seed_{seed}_classic_loss")
    logging.info(f"NN: {pickle_file}")

    discriminator = MultilayerPerceptron.get_simple_mlp(
        flatten=False,
        input_shape=input_shape,
        activation="relu",
        architecture=architecture,
        output_activation="",
        output_shape=1,
    )
    discriminator.compile(optimizer="adam", metrics=["mse"], loss="mse")
    history = discriminator.fit(
        x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0
    )

    pd.DataFrame(history.history["val_mse"]).to_csv(csv_file)
    save_pickle(pickle_file, history.history)


def run_compression_loss(
    discriminator_arch: list,
    latent_dimension: int,
    input_shape: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    pickles_dir: str,
    csvs_dir: str,
    seed: int,
):

    pickle_file = os.path.join(
        pickles_dir, f"nu_{latent_dimension}_seed_{seed}_classic_compression"
    )
    csv_file = os.path.join(
        csvs_dir, f"nu_{latent_dimension}_seed_{seed}_classic_compression.csv"
    )
    logging.info(f"UAE: {pickle_file}")
    if os.path.exists(pickle_file) and os.path.exists(csv_file):
        logging.info(f"File {csv_file} exists, returning.")
        return

    # UAE
    autoencoder = get_autoencoder(input_shape, latent_dimension)

    discriminator = MultilayerPerceptron.get_simple_mlp(
        flatten=False,
        input_shape=input_shape,
        activation="relu",
        architecture=discriminator_arch,
        output_activation="",
        output_shape=1,
    )

    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=MeanSquaredError(),
        metrics=["mse"],
    )
    discriminator.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=MeanSquaredError(),
    )

    autoencoder_history = autoencoder.fit(
        x_train, x_train, epochs=epochs, validation_data=(x_test, x_test), verbose=0
    )

    decoded_x_train = autoencoder(x_train)
    decoded_x_test = autoencoder(x_test)
    discriminator_history = discriminator.fit(
        x=decoded_x_train,
        y=y_train,
        epochs=epochs,
        validation_data=(decoded_x_test, y_test),
        verbose=0,
    )

    pd.DataFrame(discriminator_history.history["val_loss"]).to_csv(csv_file)

    save_pickle(
        pickle_file,
        {
            FileNames.cc_loss: autoencoder_history.history,
            FileNames.cc_accuracy: discriminator_history.history,
        },
    )


def run_bae_loss(
    discriminator_arch: list,
    input_shape: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    nu: int,
    epochs: int,
    csv_outputs: str,
    pickle_outputs: str,
    d_weight: float,
    n_weight: float,
    verbose: int,
    seed: int,
    training_type: int,
):

    csv_file = os.path.join(
        csv_outputs,
        f"d_weight_{d_weight}_n_weight_{n_weight}_type={training_type}_seed_{seed}_nu_{nu}_benign_compression.csv",
    )
    pickle_file = os.path.join(
        pickle_outputs,
        f"d_weight_{d_weight}_n_weight_{n_weight}_type={training_type}_seed_{seed}_nu_{nu}_benign_compression",
    )
    logging.info(f"BAE: {pickle_file}")
    encoder, decoder = get_autoencoder(input_shape, nu, split=True)
    discriminator = MultilayerPerceptron.get_simple_mlp(
        flatten=False,
        input_shape=input_shape,
        activation="relu",
        architecture=discriminator_arch,
        output_activation="",
        output_shape=1,
    )

    ae_nn = BenignCompression(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        reconstruction_loss=MeanSquaredError(),
        optimizer="adam",
        loss="mse",
        metric=["mse"],
        decompression_weight=d_weight,
        discrimination_weight=n_weight,
    )
    fits_data = ae_nn.train(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        verbose=verbose,
        training_type=training_type,
    )

    pd.DataFrame(fits_data["val_accuracy"]).to_csv(csv_file)

    save_pickle(
        pickle_file,
        {
            FileNames.bae_loss: ae_nn.compression_errors,
            FileNames.bae_accuracy: fits_data,
        },
    )


def get_bae_simulated_data_iterations() -> list:
    iterations = []
    iteration = namedtuple("Iteration", "seed d neurons nu dw nw training_type name")
    for seed in range(20):
        for d in SimulationParameters.n_features:
            nus = get_nus(d)
            for neuron in SimulationParameters.neurons:
                pickle_dir, csvs_dir = get_output_dirs(
                    data_dir, learning_rate, noise, d, neuron
                )
                for nu in nus:
                    for dw in SimulationParameters.dws:
                        for nw in SimulationParameters.nws:
                            for training_type in [0, 1, 2]:
                                pickle_file = os.path.join(
                                    pickle_dir,
                                    f"d_weight_{dw}_n_weight_{nw}_type={training_type}_seed_{seed}_nu_{nu}_benign_compression",
                                )

                                if os.path.exists(pickle_file):
                                    continue
                                iterations.append(
                                    iteration(
                                        seed,
                                        d,
                                        neuron,
                                        nu,
                                        dw,
                                        nw,
                                        training_type,
                                        "bae",
                                    )
                                )
    return iterations


def get_uae_simulated_data_iterations() -> list:
    iterations = []
    iteration = namedtuple("Iteration", "seed d neurons nu name")
    for seed in range(20):
        for d in SimulationParameters.n_features:
            nus = get_nus(d)
            for neuron in SimulationParameters.neurons:
                pickle_dir, csvs_dir = get_output_dirs(
                    data_dir, learning_rate, noise, d, neuron
                )
                for nu in nus:

                    csv_file = os.path.join(
                        csvs_dir,
                        f"nu_{nu}_seed_{seed}_classic_compression.csv",
                    )
                    if os.path.exists(csv_file):
                        continue
                    iterations.append(iteration(seed, d, neuron, nu, "uae"))
    return iterations


def get_nn_simulate_data_iterations() -> list:
    iterations = []
    iteration = namedtuple("Iteration", "seed d neurons name")
    for seed in range(20):
        for d in SimulationParameters.n_features:
            for neuron in SimulationParameters.neurons:
                pickle_dir, csvs_dir = get_output_dirs(
                    data_dir, learning_rate, noise, d, neuron
                )

                csv_file = os.path.join(
                    csvs_dir,
                    f"seed_{seed}_classic_loss.csv",
                )
                if os.path.exists(csv_file):
                    continue
                iterations.append(iteration(seed, d, neuron, "nn"))
    return iterations


def get_output_dirs(
    data_dir: str,
    learning_rate: float,
    noise: float,
    n_features: float,
    number_neurons: int,
):
    pickle_dir = os.path.join(
        data_dir,
        f"lr={learning_rate}_noise_{noise}_d={n_features}_neurons={number_neurons}",
    )
    csvs_dir = os.path.join(
        data_dir,
        f"lr={learning_rate}_noise_{noise}_d={n_features}_neurons={number_neurons}_csvs",
    )
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)
    return pickle_dir, csvs_dir


def run_nn_iteration(
    seed: int,
    n_features: int,
    number_neurons: int,
    epochs: int,
    architecture: list,
    data_dir: str,
    noise: float,
    learning_rate: float,
):
    pickle_dir, csvs_dir = get_output_dirs(
        data_dir, learning_rate, noise, n_features, number_neurons
    )

    x_train, x_test, y_train, y_test = DatasetManager.get_sigmoid_simulated_data(
        sample_size=SimulationParameters.sample_sizes[n_features],
        n_features=n_features,
        sigmoid_activation=False,
        number_neurons=number_neurons,
        noise_size=0.0,
        seed=seed,
    )

    run_classic_loss(
        architecture=architecture,
        input_shape=n_features,
        epochs=epochs,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        pickles_dir=pickle_dir,
        csvs_dir=csvs_dir,
        seed=seed,
    )


def run_uae_iteration(
    seed: int,
    n_features: int,
    number_neurons: int,
    nu: int,
    epochs: int,
    architecture: list,
    data_dir: str,
    noise: float,
    learning_rate: float,
):
    pickle_dir, csvs_dir = get_output_dirs(
        data_dir, learning_rate, noise, n_features, number_neurons
    )
    x_train, x_test, y_train, y_test = DatasetManager.get_sigmoid_simulated_data(
        sample_size=SimulationParameters.sample_sizes[n_features],
        n_features=n_features,
        sigmoid_activation=False,
        number_neurons=number_neurons,
        noise_size=0.0,
        seed=seed,
    )

    run_compression_loss(
        discriminator_arch=architecture,
        input_shape=n_features,
        latent_dimension=nu,
        epochs=epochs,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        pickles_dir=pickle_dir,
        csvs_dir=csvs_dir,
        seed=seed,
    )


def run_bae_iteration(
    seed: int,
    n_features: int,
    number_neurons: int,
    nu: int,
    dw: float,
    nw: float,
    epochs: int,
    architecture: list,
    data_dir: str,
    noise: float,
    learning_rate: float,
    training_type: int,
):
    pickle_dir, csvs_dir = get_output_dirs(
        data_dir, learning_rate, noise, n_features, number_neurons
    )

    x_train, x_test, y_train, y_test = DatasetManager.get_sigmoid_simulated_data(
        sample_size=SimulationParameters.sample_sizes[n_features],
        n_features=n_features,
        sigmoid_activation=False,
        number_neurons=number_neurons,
        noise_size=0.0,
        seed=seed,
    )

    run_bae_loss(
        architecture,
        input_shape=n_features,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        nu=nu,
        epochs=epochs,
        csv_outputs=csvs_dir,
        pickle_outputs=pickle_dir,
        d_weight=dw,
        n_weight=nw,
        verbose=0,
        seed=seed,
        training_type=training_type,
    )


def run_iteration(
    iteration,
    epochs: int,
    architecture: list,
    data_dir: str,
    noise: float,
    learning_rate: float,
):

    if iteration.name == "nn":
        run_nn_iteration(
            seed=iteration.seed,
            n_features=iteration.d,
            number_neurons=iteration.neurons,
            epochs=epochs,
            architecture=architecture,
            data_dir=data_dir,
            noise=noise,
            learning_rate=learning_rate,
        )

    if iteration.name == "uae":
        run_uae_iteration(
            seed=iteration.seed,
            n_features=iteration.d,
            number_neurons=iteration.neurons,
            nu=iteration.nu,
            epochs=epochs,
            architecture=architecture,
            data_dir=data_dir,
            noise=noise,
            learning_rate=learning_rate,
        )

    if iteration.name == "bae":
        run_bae_iteration(
            seed=iteration.seed,
            n_features=iteration.d,
            number_neurons=iteration.neurons,
            nu=iteration.nu,
            dw=iteration.dw,
            nw=iteration.nw,
            epochs=epochs,
            architecture=architecture,
            data_dir=data_dir,
            noise=noise,
            learning_rate=learning_rate,
            training_type=iteration.training_type,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc", dest="hpc", action="store_true")
    parser.set_defaults(hpc=False)

    parser.add_argument("--array-index", dest="array_index", type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    epochs = 20
    architecture = [64]
    data_dir = "data/bae_on_simulated_data_new"
    noise = 0.0
    learning_rate = 0.001

    args = parse_args()

    bae_iterations = get_bae_simulated_data_iterations()
    nn_iterations = get_nn_simulate_data_iterations()
    uae_iterations = get_uae_simulated_data_iterations()
    all_iterations = bae_iterations + nn_iterations + uae_iterations
    logging.info(f"Iterations: {len(all_iterations)}")
    if not args.hpc:
        for iteration in all_iterations:
            run_iteration(
                iteration, epochs, architecture, data_dir, noise, learning_rate
            )

    else:
        try:
            iteration = all_iterations[args.array_index]
            run_iteration(
                iteration, epochs, architecture, data_dir, noise, learning_rate
            )
        except:
            exit(0)
