import os
import numpy as np
from paper.simulated_data import get_nus, SimulationParameters
from plotting.plot_simulated_data import (
    get_simulated_data_name_dir,
    collect_results,
)
from utils.saving_results import (
    get_bae_loss_file_name,
    get_classic_loss_file_name,
    get_uae_loss_file_name,
)
from utils.saving_results import FileNames
import logging

logging.basicConfig(level=logging.INFO)


TABLE_HEADER = r"""
\begin{table}
\centering
"""

CAPTION = """\caption{}"""
TABLE_HEADER_2 = r"""
\scalebox{0.45}{
\begin{tabular}{@{}lrrrrrrrrrrrrrrrrr@{}}
\toprule
& \multicolumn{1}{c}{Model:} &
& \multicolumn{1}{c}{NN} &  
\phantom{ab} & \multicolumn{4}{c}{UAE+NN} &
\phantom{ab} & \multicolumn{4}{c}{BAE+NN} \\
\midrule
"""
TABLE_SUBHEADER = """{{\it \# Features:}}  & {} && {{\it \#Dataset Size}} && {}  &  &  &  &&  &  &  &  \\\\"""
TABLE_NUS = (
    """& \phantom{}$\\nu$ && - && {} & {} & {} & {} && {} & {} & {} & {} \\\\ """
)
TABLE_NEURONS = r"""&\textbf{Neurons}\\"""
TABLE_ROW = (
    """& \phantom{} {} && {} && {} & {} & {} & {}  && {}&  {} &  {} & {}  \\\\"""
)
TABLE_SPACE = """\\\\"""
TABLE_BOTTOM = r"""
\bottomrule
\end{tabular}
}
\end{table}
"""


def create_table(table: dict, caption: str, nus: list):

    final_table = TABLE_HEADER
    final_caption = CAPTION.format(caption)
    final_table += "\n"
    final_table += final_caption
    final_table += "\n"
    final_table += TABLE_HEADER_2
    final_table += "\n"
    for n_features in SimulationParameters.n_features:
        observations = SimulationParameters.sample_sizes[n_features]
        nus = get_nus(n_features)
        sub_header = TABLE_SUBHEADER.format(n_features, observations)
        final_table += sub_header
        final_table += "\n"
        nus_row = TABLE_NUS.format(
            "{ab}",
            nus[0],
            nus[1],
            nus[2],
            nus[3],
            nus[0],
            nus[1],
            nus[2],
            nus[3],
        )
        final_table += nus_row
        final_table += "\n"
        final_table += TABLE_NEURONS

        for neuron in SimulationParameters.neurons:
            final_table += "\n"
            final_table += TABLE_ROW.format(
                "{abc}",
                neuron,
                table[f"NN_{n_features}_{neuron}"],
                table[f"UAE+NN_{n_features}_{neuron}_{nus[0]}"],
                table[f"UAE+NN_{n_features}_{neuron}_{nus[1]}"],
                table[f"UAE+NN_{n_features}_{neuron}_{nus[2]}"],
                table[f"UAE+NN_{n_features}_{neuron}_{nus[3]}"],
                table[f"BAE+NN_{n_features}_{neuron}_{nus[0]}"],
                table[f"BAE+NN_{n_features}_{neuron}_{nus[1]}"],
                table[f"BAE+NN_{n_features}_{neuron}_{nus[2]}"],
                table[f"BAE+NN_{n_features}_{neuron}_{nus[3]}"],
            )
        final_table += "\n"
        final_table += TABLE_SPACE
    final_table += "\n"
    final_table += TABLE_BOTTOM
    return final_table


def take_mean_and_std_over_runs(data_dict: dict, get_files: list, score: str):
    """"""

    losses = []
    for file in get_files:
        results = data_dict[file]
        mse_loss = np.min(results[score])
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

    new_data_dict = {file_: data_dict[file_][FileNames.cc_loss] for file_ in get_files}
    compr_mean_loss, compr_std_loss = take_mean_and_std_over_runs(
        new_data_dict, get_files, score
    )
    return acc_mean_loss, acc_std_loss, compr_mean_loss, compr_std_loss


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

    compr_mean_loss, compr_std_loss = take_mean_and_std_over_runs(
        get_files=get_files, score=FileNames.bae_loss, data_dict=data_dict
    )
    return acc_mean_loss, acc_std_loss, compr_mean_loss, compr_std_loss


if __name__ == "__main__":

    score = "val_mse"

    noise_factor = 0.0
    lr = 0.001
    data_dir = f"data/bae_on_simulated_data_mse"
    runs = 20
    big_table = f"paper/tables/simulated_data_mse_lr={lr}_.tex"
    if os.path.exists(big_table):
        os.remove(big_table)

    for training_type in [0, 1, 2]:
        for dw in SimulationParameters.dws:
            # Fix dw, nw I want to show a table
            for nw in SimulationParameters.nws:
                table = {}
                data_dict = {}
                for d in SimulationParameters.n_features:
                    nus = get_nus(d)

                    for neurons in SimulationParameters.neurons:
                        # 1 ROW we have 1 winner to make bold
                        make_bold = None
                        min_mse = 10**10
                        name_dir = get_simulated_data_name_dir(
                            lr, noise_factor, d, neurons
                        )

                        data_dict, dws, nws = collect_results(
                            data_dict, nus, data_dir, name_dir
                        )
                        print(name_dir)
                        # *******************
                        # Average NN Loss
                        # *******************
                        file_name = get_classic_loss_file_name()
                        get_files = [file_name.format(seed) for seed in range(runs)]
                        mean_nn_loss, std_nn_loss = take_mean_and_std_over_runs(
                            data_dict=data_dict,
                            get_files=get_files,
                            score="val_mse",
                        )

                        table[
                            f"NN_{d}_{neurons}"
                        ] = f"{mean_nn_loss:.3f}+-{std_nn_loss:.3f}"
                        if mean_nn_loss < min_mse:
                            min_mse = mean_nn_loss
                            make_bold = f"NN_{d}_{neurons}"
                        for nu in nus:
                            (
                                mean_uae_loss,
                                std_uae_loss,
                                compr_mean,
                                compr_std,
                            ) = take_mean_and_std_over_uae_runs(
                                data_dict=data_dict,
                                file_name=get_uae_loss_file_name(),
                                runs=runs,
                                score="val_loss",
                                nu=nu,
                            )

                            table[
                                f"UAE+NN_{d}_{neurons}_{nu}"
                            ] = f"{mean_uae_loss:.3f}+-{std_uae_loss:.3f}"
                            if mean_uae_loss < min_mse:
                                min_mse = mean_uae_loss
                                make_bold = f"UAE+NN_{d}_{neurons}_{nu}"
                            (
                                acc_mean_loss,
                                acc_std_loss,
                                compr_mean_loss,
                                compr_std_loss,
                            ) = take_mean_and_std_over_bae_runs(
                                data_dict,
                                file_name=get_bae_loss_file_name(),
                                runs=runs,
                                score="val_accuracy",
                                nu=nu,
                                dw=dw,
                                nw=nw,
                                training_type=training_type,
                            )

                            table[
                                f"BAE+NN_{d}_{neurons}_{nu}"
                            ] = f"{acc_mean_loss:.3f}+-{acc_std_loss:.3f}"
                            if acc_mean_loss < min_mse:
                                min_mse = acc_mean_loss
                                make_bold = f"BAE+NN_{d}_{neurons}_{nu}"
                        # Make bold
                        value_to_make_bold = table[make_bold]
                        table[make_bold] = f"""\\textbf{{{value_to_make_bold}}}"""

                caption = (
                    f"{{The table below shows the $\ell^2$ loss (lower is better) for"
                    + " NN, UAE+NN, and BAE+NN models. We show the min average $\ell^2$ loss"
                    + " and standard deviation over 20 runs. For UAE+NN and BAE+NN we test"
                    + r" their accuracy for different level of $\nu$."
                    + f" Parameters: Training type={training_type}, dw={dw}, nw={nw}}}"
                )

                table = create_table(table, caption, nus)
                with open(big_table, "a") as f:
                    f.write(table)
                    f.write("\n")
