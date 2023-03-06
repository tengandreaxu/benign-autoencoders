import os
import numpy as np
import pandas as pd

from utils.saving_results import (
    get_bae_loss_file_name,
    get_classic_loss_file_name,
    get_uae_loss_file_name,
)
from utils.helper_functions import decode_pickle
from utils.saving_results import FileNames
import logging

logging.basicConfig(level=logging.INFO)


def collect_results(data_dir: str, name_dir: str) -> dict:
    data_dict = {}
    files = os.listdir(os.path.join(data_dir, name_dir))
    for filename in files:
        if "seed" not in filename:
            continue

        data_dict[filename] = decode_pickle(os.path.join(data_dir, name_dir, filename))
    return data_dict


def take_mean_and_std_over_runs(data_dict: dict, get_files: list, score: str):
    """"""

    losses = []
    for file in get_files:
        results = data_dict[file]
        mse_loss = np.max(results[score])
        losses.append(mse_loss)
    mean_loss = np.mean(losses)

    std_loss = np.std(losses)
    return mean_loss, std_loss


def take_mean_and_std_over_uae_runs(
    data_dict: dict, file_name: str, runs: int, score: str, nu: int
):

    get_files = [file_name.format(nu, seed) for seed in range(runs)]
    new_data_dict = {
        file_: data_dict[file_][FileNames.cc_accuracy] for file_ in get_files
    }

    acc_mean_loss, acc_std_loss = take_mean_and_std_over_runs(
        new_data_dict, get_files, score
    )

    return acc_mean_loss, acc_std_loss


def take_mean_and_std_over_bae_runs(
    data_dict: dict,
    file_name: str,
    runs: int,
    score: str,
    nu: int,
    dw: float,
    nw: float,
    training_type: int,
):

    get_files = [
        file_name.format(dw, nw, training_type, seed, nu) for seed in range(runs)
    ]
    new_data_dict = {
        file_: data_dict[file_][FileNames.bae_accuracy] for file_ in get_files
    }

    acc_mean_loss, acc_std_loss = take_mean_and_std_over_runs(
        new_data_dict, get_files, score
    )

    return acc_mean_loss, acc_std_loss


if __name__ == "__main__":

    lr = 0.001
    results_name = "bae_on_mnist_keras_ae_batch"
    data_dir = f"data/{results_name}"
    runs = 20

    table = []
    for noise_factor in [0.0, 0.25, 0.50, 0.75]:
        dw = 0.9
        nw = 0.1
        row = {}
        row = {"\\textit{Noise Factor}": noise_factor}
        # 1 ROW we have 1 winner to make bold
        make_bold = None
        max_accuracy = 0
        name_dir = f"noise_{noise_factor}"
        data_dict = collect_results(data_dir, name_dir)
        print(name_dir)
        # *******************
        # Average NN Loss
        # *******************
        file_name = get_classic_loss_file_name()
        get_files = [file_name.format(seed) for seed in range(runs)]
        mean_nn_loss, std_nn_loss = take_mean_and_std_over_runs(
            data_dict=data_dict,
            get_files=get_files,
            score="val_accuracy",
        )

        row[f"NN"] = f"{mean_nn_loss:.3f}+-{std_nn_loss:.3f}"
        if mean_nn_loss > max_accuracy:
            max_accuracy = mean_nn_loss
            make_bold = f"NN"

        (mean_uae_loss, std_uae_loss,) = take_mean_and_std_over_uae_runs(
            data_dict=data_dict,
            file_name=get_uae_loss_file_name(),
            runs=runs,
            score="val_accuracy",
            nu=128,
        )

        row[f"UAE+NN"] = f"{mean_uae_loss:.3f}+-{std_uae_loss:.3f}"
        if mean_uae_loss > max_accuracy:
            max_accuracy = mean_uae_loss
            make_bold = f"UAE+NN"

        for training_type in [0, 1, 2]:
            (acc_mean_loss, acc_std_loss,) = take_mean_and_std_over_bae_runs(
                data_dict,
                file_name=get_bae_loss_file_name(),
                runs=runs,
                score="val_accuracy",
                nu=128,
                dw=dw,
                nw=nw,
                training_type=training_type,
            )
            row[f"BAE+NN({training_type})"] = f"{acc_mean_loss:.3f}+-{acc_std_loss:.3f}"
            if acc_mean_loss > max_accuracy:
                max_accuracy = acc_mean_loss
                make_bold = f"BAE+NN({training_type})"

        # Make bold
        value_to_make_bold = row[make_bold]
        row[make_bold] = f"""\\textbf{{{value_to_make_bold}}}"""
        table.append(row)

    file_name = f"paper/tables/{results_name}_lr=0.001_nw={nw}_dw={dw}_.tex"
    df = pd.DataFrame(table)
    caption = (
        f"{{The table below shows the test accuracy (higher is better) for"
        + " NN, UAE+NN, and BAE+NN models trained on MNIST. We show the mean"
        + " and the standard deviation of the best model's performance over 20 runs."
        + f" Parameters: dw={dw}, nw={nw}}}"
    )
    df.to_latex(
        file_name,
        label="table:mnist_results",
        escape=False,
        index=False,
        caption=caption,
    )
