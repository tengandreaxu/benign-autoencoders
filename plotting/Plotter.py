import numpy as np
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
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}
pylab.rcParams.update(params)


@dataclass
class Plotter:
    def __init__(self):
        self.logger = logging.getLogger("Plotter")
        self.colors = ["black", "brown"]
        self.line_styles = ["solid", "dashed"]

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
        vline: Optional[float] = 0,
        vline_label: Optional[str] = "",
    ):

        plt.plot(x, y, label=label, color="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if vline > 0:
            plt.axvline(vline, linestyle="dashed", color="red", label=vline_label)
        plt.legend(ncol=ncol)
        plt.grid(True)
        plt.tight_layout()
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
    ):
        plt.figure(figsize=(15, 10))
        for y, color, line_style in zip(ys, self.colors, self.line_styles):
            lists, label_ = y
            plt.plot(xs, lists, label=label_, color=color, linestyle=line_style)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(ncol=ncol)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_multiple_curves_lists_next_to(
        self,
        ys1: list,  # [Tuple[str, list]],
        xs1: list,
        ys2: list,  # [Tuple[str, list]],
        xs2: list,
        y1label: str,
        x1label: str,
        y2label: str,
        x2label: str,
        file_name: str,
        suptitle: str,
        subtitle1: str,
        subtitle2: str,
        ncol: int = 1,
        show: Optional[bool] = False,
    ):

        fig, axs = plt.subplots(1, 2, figsize=(15, 13))

        for y in ys1:

            lists, label_ = y
            axs[0].plot(xs1, lists, label=label_)

        for y in ys2:
            lists, label_ = y

            axs[1].plot(xs2, lists, label=label_)

        axs[0].set_xlabel(x1label)
        axs[0].set_ylabel(y1label)

        axs[1].set_xlabel(x2label)
        axs[1].set_ylabel(y2label)

        axs[0].set_title(subtitle1)
        axs[1].set_title(subtitle2)

        axs[0].legend(ncol=ncol)
        axs[1].legend(ncol=ncol)

        axs[0].grid(True)
        axs[1].grid(True)

        fig.suptitle(suptitle, size="xx-large", weight="bold")

        plt.grid(True)
        # fig.tight_layout()

        if show:
            plt.savefig(file_name)
            plt.show()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_four_curves(
        self,
        ys1: list,  # [Tuple[str, list]],
        xs1: list,
        ys2: list,  # [Tuple[str, list]],
        xs2: list,
        ys3: list,  # [Tuple[str, list]],
        xs3: list,
        ys4: list,  # [Tuple[str, list]],
        xs4: list,
        y1label: str,
        x1label: str,
        y2label: str,
        x2label: str,
        y3label: str,
        x3label: str,
        y4label: str,
        x4label: str,
        file_name: str,
        suptitle: str,
        subtitle1: str,
        subtitle2: str,
        subtitle3: str,
        subtitle4: str,
        ncol: int = 1,
        show: Optional[bool] = False,
    ):

        fig, axs = plt.subplots(2, 2, figsize=(28, 23), sharex=True)

        for y in ys1:
            lists, label_ = y
            axs[0, 0].plot(xs1, lists, label=label_)

        for y in ys2:
            lists, label_ = y
            axs[1, 0].plot(xs2, lists, label=label_)

        for y in ys3:
            lists, label_ = y
            axs[0, 1].plot(xs3, lists, label=label_)

        for y in ys4:
            lists, label_ = y
            axs[1, 1].plot(xs4, lists, label=label_)

        axs[0, 0].set_ylabel(y1label, fontdict={"size": 20})
        axs[1, 0].set_ylabel(y2label, fontdict={"size": 20})
        axs[0, 1].set_ylabel(y3label, fontdict={"size": 20})
        axs[1, 1].set_ylabel(y4label, fontdict={"size": 20})

        axs[0, 0].set_xlabel(x1label, fontdict={"size": 20})
        axs[1, 0].set_xlabel(x2label, fontdict={"size": 20})
        axs[0, 1].set_xlabel(x3label, fontdict={"size": 20})
        axs[1, 1].set_xlabel(x4label, fontdict={"size": 20})

        axs[0, 0].set_title(subtitle1, fontdict={"size": 20})
        axs[1, 0].set_title(subtitle2, fontdict={"size": 20})
        axs[0, 1].set_title(subtitle3, fontdict={"size": 20})
        axs[1, 1].set_title(subtitle4, fontdict={"size": 20})

        axs[0, 0].legend(ncol=ncol, fontsize=18)
        axs[1, 0].legend(ncol=ncol, fontsize=18)
        axs[0, 1].legend(ncol=ncol, fontsize=18)
        axs[1, 1].legend(ncol=ncol, fontsize=18)

        axs[0, 0].tick_params(labelsize=18)
        axs[1, 0].tick_params(labelsize=18)
        axs[0, 1].tick_params(labelsize=18)
        axs[1, 1].tick_params(labelsize=18)

        axs[0, 0].grid(True)
        axs[1, 0].grid(True)
        axs[0, 1].grid(True)
        axs[1, 1].grid(True)

        fig.suptitle(suptitle, size=24, weight="bold")

        # plt.grid(True)
        # fig.tight_layout()

        if show:
            plt.savefig(file_name)
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
