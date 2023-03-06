import pickle
import os
from collections import namedtuple
from itertools import product


def make_pairs_named_tuple(**data):
    """HPC nodes' task allocation"""
    list_of_labels = [key for key in data.keys()]
    list_of_data_values = [val for val in data.values()]
    all_the_products = list(product(*list_of_data_values))
    Iteration = namedtuple("Iteration", list_of_labels, defaults=(None,))
    Iterations = [
        Iteration(*single_data_point) for single_data_point in all_the_products
    ]
    return Iterations


def save_pickle(name_file: str, variables):
    """Saves the object as pickle"""
    with open(name_file, "wb") as handle:
        pickle.dump(variables, handle, protocol=4)


def decode_pickle(name_file):
    """Loads back the pickle"""
    with open(name_file, "rb") as handle:
        breakpoint()
        pickled_dict = pickle.load(handle)
        return pickled_dict


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


def filter_iterations(iterations: list, save_dir: str):
    output = []

    for iteration in iterations:
        noise_factor = iteration.noise_factor
        pickle_output = os.path.join(save_dir, f"noise_{noise_factor}")
        seed = iteration.seed
        if iteration.loss == "classic_loss":
            pickle_file = os.path.join(pickle_output, f"seed_{seed}_classic_loss")

            if os.path.exists(pickle_file):
                continue
        if iteration.loss == "compression_loss":
            nu = iteration.latent_dim
            if nu == 0:
                nu = 128
            pickle_file = os.path.join(
                pickle_output, f"nu_{nu}_seed_{seed}_classic_compression"
            )
            if os.path.exists(pickle_file):
                continue
        if iteration.loss == "benign_loss":
            dw = iteration.d_weight
            nw = iteration.n_weight
            training_type = iteration.training_type
            nu = iteration.latent_dim
            if nu == 0:
                nu = 128
            pickle_file = os.path.join(
                pickle_output,
                f"d_weight_{dw}_n_weight_{nw}_type={training_type}_seed_{seed}_nu_{nu}_benign_compression",
            )
            if os.path.exists(pickle_file):
                continue

        output.append(iteration)
    return output
