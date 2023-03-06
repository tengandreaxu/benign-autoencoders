import os
import logging

logging.basicConfig(level=logging.INFO)


class FileNames:
    """
    cc: ClassicCompression
    bae: BenignAutoEncoder
    """

    cc_loss = "fit_COMPRESSION_CLASS_COMPRESSION"
    cc_accuracy = "fit_DISCRIMINATION_CLASS_COMPRESSION"
    bae_loss = "fit_COMPRESSION_BENIGN_COMPRESSION"
    bae_accuracy = "fit_DISCRIMINATION_BENIGN_COMPRESSION"


def get_simulated_data_name_dir(lr: float, noise_factor: float, d: float, neurons: int):
    return f"lr={lr}_noise_{noise_factor}_d={d}_neurons={neurons}"


def get_classic_loss_file_name():
    return "seed_{}_classic_loss"


def get_uae_loss_file_name():
    return "nu_{}_seed_{}_classic_compression"


def get_bae_loss_file_name():

    return "d_weight_{}_n_weight_{}_type={}_seed_{}_nu_{}_benign_compression"


def get_bae_sub_file_name():

    return "d_weight_{}_n_weight_{}_type={}_seed_{}_nu_{}_bae_sub"


def are_results_there(csv_file: str, pickle_file: str) -> bool:
    exists = os.path.exists(csv_file) and os.path.exists(pickle_file)
    if exists:
        logging.info(f"File {csv_file} exists, returning.")
    return exists
