import os
import numpy as np
import pandas as pd
from plotting.Plotter import Plotter
from experiments.mnist.ushape_classification import Z_DIMS

if __name__ == "__main__":
    dataset = "mnist"
    noise_factor = 1
    epochs = 20
    original_dimension = 28 * 28
    output_dir = f"results/ushape_classification/{dataset}/joint_noise={noise_factor}_epochs={epochs}"
    z_dims = Z_DIMS
    bae_loss = []
    bae_rec_loss = []
    best_acc = 0
    best_rec_loss = np.inf
    for z_dim in z_dims:
        df = pd.read_csv(os.path.join(output_dir, f"z={z_dim}", "history.csv"))
        acc = df.test_accuracy.max()
        rec_loss = df.test_rec_loss.min()
        bae_rec_loss.append(rec_loss)
        bae_loss.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_z = z_dim
            print(f"Acc:\t{best_acc:.3f}\tZ:{z_dim}")
        if rec_loss < best_rec_loss:
            best_rec_loss = rec_loss
            best_z_rec = z_dim
            print(f"Rec:\t{best_rec_loss:.3f}\tZ:{z_dim}")

    for format in [".png", ".pdf"]:
        plotter = Plotter()
        plotter.plot_single_curve(
            x=z_dims,
            y=bae_loss,
            ylabel="Accuracy (%)",
            xlabel=r"$\nu$",
            file_name=os.path.join(output_dir, f"accuracy{format}"),
            vlines=[best_z, original_dimension],
            vline_labels=[
                r"$\nu^{}\, =\, {}$".format("*", best_z),
                f"Original Dimension = {original_dimension}",
            ],
            title="",
            grid=False,
            label=r"$W(\mathcal{D}_{\theta}(\mathcal{E}_{\phi}(\tilde{x})))$",
        )

        plotter.plot_single_curve(
            x=z_dims,
            y=bae_rec_loss,
            ylabel="Reconstruction Loss",
            xlabel=r"$\nu$",
            file_name=os.path.join(output_dir, f"reconstruction{format}"),
            vlines=[best_z_rec, 784],
            vline_labels=[
                r"$\nu^{} = {}$".format("*", best_z_rec),
                f"Original Dimension = {original_dimension}",
            ],
            title="",
            grid=False,
            label=r"$W(\mathcal{D}_{\theta}(\mathcal{E}_{\phi}(\tilde{x})))$",
        )
