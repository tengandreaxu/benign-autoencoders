import os
import argparse
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from datasets.DatasetsManager import DatasetManager

from keras import layers, losses, optimizers, metrics
from keras.losses import BinaryCrossentropy

from models.BenignCompression import BenignCompression
from models.autoencoders.KerasMNISTAE import KerasMNISTAE
from models.autoencoders.KaggleMNISTAE import KaggleMNISTAE
from utils.helper_functions import save_pickle
from utils.saving_results import FileNames, are_results_there
from models.discriminators.mnist import Discriminator
from utils.helper_functions import touch, filter_iterations, make_pairs_named_tuple


def run_compression_loss(
    learning_rate,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    dataset_name: str,
    csv_output: str,
    pickle_output: str,
    seed: int,
):

    pickle_file = os.path.join(
        pickle_output,
        f"nu_128_seed_{seed}_classic_compression",
    )
    csv_file = os.path.join(
        csv_output,
        f"nu_128_seed_{seed}_classic_compression.csv",
    )
    if are_results_there(csv_file, pickle_file):
        return
    touch(pickle_file)
    touch(csv_file)
    if dataset_name == "mnist":
        print(f"Keras Autoencoder")
        autoencoder = KerasMNISTAE()
    else:
        print(f"Kaggle Autoencoder")
        autoencoder = KaggleMNISTAE()
    discriminator = Discriminator()

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
    )

    discriminator.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    compression_loss = autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    discrimination_loss = discriminator.fit(
        autoencoder(x_train),
        y_train,
        epochs=epochs,
        shuffle=True,
        validation_data=(autoencoder(x_test), y_test),
    )
    pd.DataFrame(discrimination_loss.history["val_accuracy"]).to_csv(csv_file)

    save_pickle(
        pickle_file,
        {
            FileNames.cc_loss: compression_loss.history,
            FileNames.cc_accuracy: discrimination_loss.history,
        },
    )


def run_bae_loss(
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    dw,
    nw,
    training_type,
    dataset_name: str,
    csv_output: str,
    pickle_output: str,
    seed: int,
):
    csv_file = os.path.join(
        csv_output,
        f"d_weight_{dw}_n_weight_{nw}_type={training_type}_seed_{seed}_nu_128_benign_compression.csv",
    )
    pickle_file = os.path.join(
        pickle_output,
        f"d_weight_{dw}_n_weight_{nw}_type={training_type}_seed_{seed}_nu_128_benign_compression",
    )
    if are_results_there(csv_file, pickle_file):
        return
    touch(pickle_file)
    touch(csv_file)

    if dataset_name == "mnist":
        print(f"Keras Autoencoder")
        decoder = KerasMNISTAE.get_decoder()
        encoder = KerasMNISTAE.get_encoder()
    else:
        print(f"Kaggle Autoencoder")
        decoder = KaggleMNISTAE.get_decoder()
        encoder = KaggleMNISTAE.get_encoder()

    discriminator = Discriminator()

    ae_nn = BenignCompression(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        optimizer=optimizers.Adam(),
        reconstruction_loss=BinaryCrossentropy(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metric=["accuracy"],
        decompression_weight=dw,
        discrimination_weight=nw,
    )

    fits_data = ae_nn.train(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
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


def run_classic_loss(
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    csv_output: str,
    pickle_output: str,
    seed: int,
):
    csv_file = os.path.join(csv_output, f"seed_{seed}_classic_loss.csv")
    pickle_file = os.path.join(pickle_output, f"seed_{seed}_classic_loss")
    if are_results_there(csv_file, pickle_file):
        return
    discriminator = Discriminator()
    discriminator.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    classic_loss = discriminator.fit(
        x_train,
        y_train,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_test, y_test),
    )

    pd.DataFrame(classic_loss.history["val_accuracy"]).to_csv(csv_file)

    save_pickle(
        pickle_file,
        classic_loss.history,
    )


def run_experiment(which, single_iteration_data, epochs):
    seed = single_iteration_data.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dataset_name = single_iteration_data.dataset_name
    noise_factor = single_iteration_data.noise_factor
    save_dir = single_iteration_data.directory
    os.makedirs(save_dir, exist_ok=True)
    csv_output = os.path.join(save_dir, f"noise_{noise_factor}_csvs")
    pickle_output = os.path.join(save_dir, f"noise_{noise_factor}")
    os.makedirs(csv_output, exist_ok=True)
    os.makedirs(pickle_output, exist_ok=True)

    dm = DatasetManager()

    (x_train, y_train), (x_test, y_test) = dm.get_nist_dataset(
        dataset_name, noise_factor=noise_factor, seed=seed
    )

    learning_rate = single_iteration_data.learning_rate
    if "classic_loss" in which:

        run_classic_loss(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            csv_output=csv_output,
            pickle_output=pickle_output,
            seed=seed,
        )
    if "compression_loss" in which:
        nu = single_iteration_data.latent_dim
        run_compression_loss(
            learning_rate,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            dataset_name,
            csv_output,
            pickle_output,
            seed,
        )

    if "benign_loss" in which:
        nu = single_iteration_data.latent_dim
        run_bae_loss(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            single_iteration_data.d_weight,
            single_iteration_data.n_weight,
            single_iteration_data.training_type,
            dataset_name,
            csv_output,
            pickle_output,
            seed,
        )


def run_trainings(iterations_data: list, epochs: int):
    for single_iteration_data in iterations_data:
        start_time = time.time()
        print("Running simulation:", single_iteration_data)
        print(single_iteration_data, flush=True)
        run_experiment(
            which=[single_iteration_data.loss],
            single_iteration_data=single_iteration_data,
            epochs=epochs,
        )

        end_time = time.time()
        print(end_time - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", action="append", required=True, type=float, dest="noise_factors"
    )
    parser.add_argument(
        "-dw", action="append", required=True, type=float, dest="decompr_factors"
    )
    parser.add_argument(
        "-nw", action="append", required=True, type=float, dest="n_factors"
    )

    parser.set_defaults(latent_dimensions=[0])
    parser.add_argument(
        "-lr", action="append", required=True, type=float, dest="learning_rates"
    )
    parser.add_argument(
        "-tp", action="append", required=True, type=int, dest="training_types"
    )

    parser.add_argument("-idx", required=False, type=int, dest="array_index")
    parser.add_argument("-e", required=True, type=int, dest="epochs")
    parser.add_argument("-hpc", dest="hpc", action="store_true")
    parser.set_defaults(hpc=False)

    parser.add_argument("--dataset", required=True, type=str, dest="dataset")

    args = parser.parse_args()
    start_time = time.time()
    seeds = range(20)
    dataset = args.dataset
    autoencoder_name = "keras_ae" if dataset == "mnist" else "kaggle_ae"
    save_dir = os.path.join("data/", f"bae_on_{dataset}_{autoencoder_name}")

    classic_loss_iterations_data = make_pairs_named_tuple(
        loss=["classic_loss"],
        noise_factor=args.noise_factors,
        learning_rate=args.learning_rates,
        seed=seeds,
        dataset_name=[dataset],
        directory=[save_dir],
    )
    compress_loss_iterations_data = make_pairs_named_tuple(
        loss=["compression_loss"],
        noise_factor=args.noise_factors,
        latent_dim=args.latent_dimensions,
        learning_rate=args.learning_rates,
        seed=seeds,
        dataset_name=[dataset],
        directory=[save_dir],
    )

    benign_loss_iterations_data = make_pairs_named_tuple(
        loss=["benign_loss"],
        noise_factor=args.noise_factors,
        d_weight=args.decompr_factors,
        n_weight=args.n_factors,
        latent_dim=args.latent_dimensions,
        learning_rate=args.learning_rates,
        training_type=args.training_types,
        seed=seeds,
        dataset_name=[dataset],
        directory=[save_dir],
    )

    iterations_data_all = (
        compress_loss_iterations_data
        + benign_loss_iterations_data
        + classic_loss_iterations_data
    )

    iterations_data = filter_iterations(iterations_data_all, save_dir)

    breakpoint()
    print(f"Iterations: {len(iterations_data)}")

    if args.hpc:
        try:
            single_iteration_data = iterations_data[args.array_index]
        except:
            exit(0)
        print("Running simulation:", args.array_index)
        print(single_iteration_data, flush=True)
        run_experiment(
            which=[single_iteration_data.loss],
            single_iteration_data=single_iteration_data,
            epochs=args.epochs,
        )

        end_time = time.time()
        print(end_time - start_time)
    else:
        run_trainings(
            iterations_data=iterations_data,
            epochs=args.epochs,
        )
