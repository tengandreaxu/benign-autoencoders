import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)


# *********************
# Plots Palette and Styles
# *********************
params = {
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 12,
}
pylab.rcParams.update(params)


@dataclass
class Plotter:
    def __init__(self):
        self.logger = logging.getLogger("Plotter")
        self.colors = ["black", "brown"]
        self.line_styles = ["solid", "dashed"]
        self.vline_colors = ["green", "red"]

    def plot_images_set(self, images: np.ndarray, file_name: str):
        # Create a 3x3 grid of subplots
        fig, axs = plt.subplots(3, 3)

        # Flatten the subplots into a 1D array
        axs = axs.flatten()

        # Loop through the images and plot them on the subplots
        for i, image in enumerate(images):
            axs[i].imshow(image)
            axs[i].axis("off")
        fig.savefig(file_name)
        plt.close()

    def show_images(self, images, title: str, file_name: str):
        """Shows the provided images as sub-pictures in a square"""

        # Converting images to CPU numpy arrays
        import torch

        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx][0], cmap="gray")
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Showing the figure
        plt.savefig(file_name)

    def plot_single_curve(
        self,
        x: list,
        y: list,
        xlabel: str,
        ylabel: str,
        file_name: str,
        title: str,
        label: str,
        ncol: int = 1,
        grid: bool = True,
        vlines: Optional[list] = [],
        vline_labels: Optional[list] = [],
        hline: Optional[float] = 0,
        hline_label: Optional[str] = "",
        legend: bool = True,
    ):
        plt.plot(x, y, label=label, color="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(vlines) > 0:
            for i, vline in enumerate(vlines):
                plt.axvline(
                    vline,
                    linestyle="dashed",
                    color=self.vline_colors[i],
                    label=vline_labels[i],
                )
        if hline > 0:
            plt.axhline(hline, linestyle="dashed", color="red", label=hline_label)

        plt.grid(grid)
        plt.tight_layout()
        if legend:
            plt.legend()
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_curves_lists(
        self,
        ys: list,  # [Tuple[str, list]],
        xs: list,
        ylabel: str,
        xlabel: str,
        file_name: str,
        title: str,
        ncol: int = 1,
        show: Optional[bool] = False,
        grid: Optional[bool] = False,
        colors: Optional[list] = ["black", "brown"],
        line_styles: Optional[list] = ["solid", "dashed"],
    ):
        for y, color, line_style in zip(ys, colors, line_styles):
            lists, label_ = y
            plt.plot(xs, lists, label=label_, color=color, linestyle=line_style)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(ncol=ncol)
        plt.title(title)
        plt.grid(grid)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_series_of_figures(
        self, figures: np.ndarray, width: int, height: int, file_name: str, title: str
    ):
        plt.figure(figsize=(10, 10))
        for i in range(figures.shape[0]):
            ax = plt.subplot(width, height, i + 1)
            plt.imshow(figures[i])
            plt.axis("off")
        plt.suptitle(title)
        plt.savefig(file_name)
        plt.close()
