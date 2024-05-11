#!/usr/bin/env python3

import copy
import sys
from typing import Literal
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

from matplotlib.axes import Axes
from matplotlib import colormaps as CM

cmap = CM.get_cmap("Dark2")
tab20c = CM.get_cmap("tab20c")
tab20b = CM.get_cmap("tab20b")

PLOTTING_DATA = {
    "node": {
        "mcnode26": {
            "artist": Line2D(
                [], [], linewidth=0, color="grey", marker="o", label="mcnode26"
            ),
            "artist2": Rectangle(
                (0, 0), 0, 0, color=tab20b(18), linewidth=0, label="mcnode26"
            ),
            "marker": {
                "shape": "o",
                "size": 4,
            },
        },
        "mcnode44": {
            "artist": Line2D(
                [], [], linewidth=0, color="grey", marker="^", label="mcnode44"
            ),
            "artist2": Rectangle(
                (0, 0), 0, 0, color=tab20b(10), linewidth=0, label="mcnode44"
            ),
            "marker": {
                "shape": "^",
                "size": 4,
            },
        },
    },
    "cores": {
        "(0, 7)": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="single",
                color=cmap(0),
                linestyle="dotted",
            ),
        },
        "(12, 19)": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="bal.",
                color=cmap(5),
                linestyle="dotted",
            ),
        },
        "(9, 16)": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="unb.",
                color=cmap(1),
                linestyle="dotted",
            ),
        },
        "(28, 35)": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="bal.",
                color=cmap(5),
                linestyle="dotted",
            ),
        },
        "(25, 32)": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="unb.",
                color=cmap(1),
                linestyle="dotted",
            ),
        },
    },
    "wl": {
        "ffmpeg": {
            "artist": Line2D([], [], linewidth=1, color=tab20c(12), label="VT"),
            "compute": {
                "color": tab20c(19),
                "range": (1.1, 7.9),
            },
        },
        "matmul": {
            "artist": Line2D([], [], linewidth=1, color=tab20c(8), label="MM"),
            "compute": {
                "color": tab20c(19),
                "range": (2.7, 4.2),
            },
        },
    },
}

LegendKeyword = Literal["node", "cores", "freq"] | None


def set_legend(ax: Axes, legend: LegendKeyword) -> Axes:
    lgnd_args = dict(framealpha=0.8, handlelength=1, handleheight=0.8)
    match legend:
        case None:
            pass
        case "node":
            ax.legend(
                [x["artist"] for x in PLOTTING_DATA["node"].values()],
                [x["artist"].get_label() for x in PLOTTING_DATA["node"].values()],
                ncol=1,
                loc="upper left",
                fontsize=9,
                markerscale=0.7,
                **lgnd_args,
            )
        case "cores":
            artists = [x["artist"] for x in PLOTTING_DATA["cores"].values()]
            artists = artists[:3] + copy.deepcopy(artists[2:3])
            artists[-1].set_linestyle("dashed")
            lbls = [x["artist"].get_label() for x in PLOTTING_DATA["cores"].values()]
            lbls = lbls[:3] + lbls[2:3]
            ax.legend(artists, lbls, ncol=2, fontsize=9, loc="upper left", **lgnd_args)
        case "freq":
            artists = [x["artist"] for x in PLOTTING_DATA["cores"].values()]
            artists = (
                copy.deepcopy(artists[2:3])
                + [Line2D([], [], color="grey", linestyle="dotted")]
                + [x["artist2"] for x in PLOTTING_DATA["node"].values()]
            )
            artists[0].set_linestyle("dashed")
            lbls = [x["artist"].get_label() for x in PLOTTING_DATA["cores"].values()]
            lbls = (
                lbls[2:3]
                + ["rest"]
                + [x["artist2"].get_label() for x in PLOTTING_DATA["node"].values()]
            )
            ax.legend(artists, lbls, ncol=1, fontsize=9, loc="upper left", **lgnd_args)
        case _:
            assert False
    return ax


def draw_compute_span(ax: Axes, wl: str) -> Axes:
    compute_range = PLOTTING_DATA["wl"][wl]["compute"]["range"]
    ax.axvspan(
        xmin=compute_range[0],
        xmax=compute_range[1],
        color=PLOTTING_DATA["wl"][wl]["compute"]["color"],
        linewidth=0,
        alpha=0.5,
    )
    return ax


def plot_pwr(df: pd.DataFrame, ax: Axes, legend: LegendKeyword) -> Axes:
    ax.grid(visible=True, axis="y", linestyle="dotted")

    wl_set = set()
    for _, ((N, WL, C, _), df) in enumerate(  # pyright: ignore
        df.groupby(["node", "wl", "cores", "run#"])
    ):
        wl_set.add(WL)
        unb_val = "(25, 32)"
        args = dict(
            linewidth=1.2,
            linestyle=(
                "dashed"
                if C == unb_val
                else PLOTTING_DATA["cores"][C]["artist"].get_linestyle()
            ),
            marker=PLOTTING_DATA["node"][N]["marker"]["shape"],
            markersize=PLOTTING_DATA["node"][N]["marker"]["size"],
            markevery=8,
            color=PLOTTING_DATA["cores"][C]["artist"].get_color(),
            alpha=1 if C == unb_val else 0.5,
            zorder=2 if C != unb_val else 3,
        )
        print(args, file=sys.stderr)
        ax.plot(
            df["at"].rolling(window=2).mean(),
            df.pwr["mean"].rolling(window=2).mean(),
            **args,  # pyright: ignore
        )

    for wl in wl_set:
        ax = draw_compute_span(ax, wl)
    ax = set_legend(ax, legend)
    ax.set_yticks(np.arange(ax.get_yticks()[0], 175 + 1, 25))
    return ax


def plot_freq(df: pd.DataFrame, ax: Axes, legend: LegendKeyword) -> Axes:
    ax.grid(visible=True, axis="y", linestyle="dotted")

    wl_set = set()
    for _, ((N, WL, C, _), df) in enumerate(  # pyright: ignore
        df.groupby(["node", "wl", "cores", "run#"])
    ):
        assert N in PLOTTING_DATA["node"]
        unb_val = "(25, 32)"
        wl_set.add(WL)
        args = dict(
            linewidth=1.2,
            linestyle=(
                "dashed"
                if C == unb_val
                else PLOTTING_DATA["cores"][C]["artist"].get_linestyle()
            ),
            color=(
                PLOTTING_DATA["cores"][C]["artist"].get_color()
                if C == unb_val
                else PLOTTING_DATA["node"][N]["artist2"].get_facecolor()
            ),
            alpha=1 if C == unb_val else 0.5,
            zorder=2 if C != unb_val else 3,
        )
        print(args, file=sys.stderr)
        ax.plot(
            df["at"].rolling(window=8).mean(),
            df.freq.rolling(window=8).mean() / 1e3,
            **args,  # pyright: ignore
        )
    for wl in wl_set:
        ax = draw_compute_span(ax, wl)
    ax.set_yticks([0, 3, 10])
    ax = set_legend(ax, legend)
    return ax


def main():
    plt.rcParams["font.family"] = "DejaVu Serif"

    df = pd.read_csv("processed/numa.csv")
    df = df[(df["run#"] == 1) | (df["run#"] == 1)]
    assert isinstance(df, pd.DataFrame)

    df_pwr = (
        df.groupby(["node", "wl", "cores", "run#", "at"])[
            ["duration_time", "power/energy-pkg/", "pwr"]
        ]
        .agg(["mean", "std"])
        .reset_index(drop=False)
    )
    assert isinstance(df_pwr, pd.DataFrame)
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharex="row",
        sharey="col",
        figsize=(5, 4.5),
        tight_layout=True,
    )
    ax[0, 0].set_title("Power (W)", size="medium")
    ax[0, 1].set_title("Frequency (GHz)", size="medium")

    ax[0, 0] = plot_pwr(
        df_pwr[df_pwr["wl"] == "matmul"],  # pyright: ignore
        ax[0, 0],
        legend="node",
    )
    ax[0, 0].text(4, 170, "MM", fontsize=12, color="black")
    ax[1, 0] = plot_pwr(
        df_pwr[df_pwr["wl"] == "ffmpeg"],  # pyright: ignore
        ax[1, 0],
        legend="cores",
    )
    ax[1, 0].text(7.7, 15, "VT", fontsize=12, color="black")
    ax[1, 0].set_xlabel("Time (s)")
    ax[0, 1] = plot_freq(
        df[df["wl"] == "matmul"],  # pyright: ignore
        ax[0, 1],
        legend="freq",
    )
    ax[1, 1] = plot_freq(
        df[df["wl"] == "ffmpeg"],  # pyright: ignore
        ax[1, 1],
        legend="freq",
    )
    ax[1, 1].set_xlabel("Time (s)")
    ax[0, 0].set_xticks(np.arange(0, 5 + 1, 1))
    with pdf.PdfPages("figure/plot_numa.pdf") as f:
        f.savefig(fig)


if __name__ == "__main__":
    main()
