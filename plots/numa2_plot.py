#!/usr/bin/env python3

import sys
from typing import Literal
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

from matplotlib.axes import Axes
from matplotlib import colormaps as CM

cmap = CM.get_cmap("Dark2")
tab20c = CM.get_cmap("tab20c")

PLOTTING_DATA = {
    "node": {
        "mcnode26": {
            "artist": Line2D(
                [], [], linewidth=0, color="grey", marker="o", label="mcnode26"
            ),
            "marker": {
                "shape": "o",
                "size": 4,
            },
        },
    },
    "cores": {
        "single": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="single",
                color=cmap(0),
                linestyle="dotted",
            ),
        },
        "balanced": {
            "artist": Line2D(
                [],
                [],
                linewidth=1,
                label="bal.",
                color=cmap(5),
                linestyle="dotted",
            ),
        },
        "unbalanced": {
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
        "matmul": {
            "compute": {
                "color": tab20c(19),
                "range": (5.4, 10.5),
            },
        },
    },
}

LegendKeyword = Literal["cores"] | None


def set_legend(ax: Axes, legend: LegendKeyword) -> Axes:
    lgnd_args = dict(framealpha=0.8, handlelength=1, handleheight=0.8)
    match legend:
        case None:
            pass
        case "cores":
            artists = [x["artist"] for x in PLOTTING_DATA["cores"].values()]
            artists = artists[:3]
            artists[-1].set_linestyle("dashed")
            lbls = [x["artist"].get_label() for x in PLOTTING_DATA["cores"].values()]
            lbls = lbls[:3]
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
        zorder=1,
    )
    return ax


def plot_pwr(df: pd.DataFrame, ax: Axes, legend: LegendKeyword) -> Axes:
    ax.grid(visible=True, axis="y", linestyle="dotted")
    ax.set_title("Power (W)", size="medium")

    wl_set = set()
    for (_, AFF, WL, _, _), df in df.groupby(  # pyright: ignore
        ["node", "affinity", "wl", "cores", "run#"]
    ):
        wl_set.add(WL)
        args = dict(
            linewidth=1.3,
            linestyle=(
                "dashed"
                if AFF == "unbalanced"
                else PLOTTING_DATA["cores"][AFF]["artist"].get_linestyle()
            ),
            color=PLOTTING_DATA["cores"][AFF]["artist"].get_color(),
            alpha=1 if AFF == "unbalanced" else 0.8,
            zorder=2 if AFF == "unbalanced" else 3,
        )
        print(args, file=sys.stderr)
        ax.plot(
            df["at"].rolling(window=2).mean(),
            df.pwr["mean"].rolling(window=2).mean(),
            **args,  # pyright: ignore
        )

    for wl in wl_set:
        ax = draw_compute_span(ax, wl)
    ax.set_yticks(np.arange(ax.get_yticks()[0], ax.get_yticks()[-1] + 1, 30))
    ax.set_xticks(np.arange(ax.get_xticks()[0], ax.get_xticks()[-1] + 1, 2.5))
    ax = set_legend(ax, legend)
    return ax


def plot_cs(df: pd.DataFrame, ax: Axes, legend: LegendKeyword) -> Axes:
    ax.grid(visible=True, axis="y", linestyle="dotted")
    ax.set_title("# cores not in C1/C2", size="medium")

    wl_set = set()
    for (_, AFF, WL, _, _), df in df.groupby(  # pyright: ignore
        ["node", "affinity", "wl", "cores", "run#"]
    ):
        wl_set.add(WL)
        args = dict(
            linewidth=1.3,
            linestyle=(
                "dashed"
                if AFF == "unbalanced"
                else PLOTTING_DATA["cores"][AFF]["artist"].get_linestyle()
            ),
            color=PLOTTING_DATA["cores"][AFF]["artist"].get_color(),
            alpha=1 if AFF == "unbalanced" else 0.8,
            zorder=2 if AFF == "unbalanced" else 3,
        )
        print(args, file=sys.stderr)
        ax.plot(
            df["at"],
            df["corecount"] - df["in_cstate"],
            **args,  # pyright: ignore
        )
    for wl in wl_set:
        ax = draw_compute_span(ax, wl)
    ax.set_yticks([1, 16, 20])
    ax = set_legend(ax, legend)
    return ax


def plot_freq(df: pd.DataFrame, ax: Axes, legend: LegendKeyword) -> Axes:
    ax.grid(visible=True, axis="y", linestyle="dotted")
    ax.set_title("Freq. of cores not in C1/C2 (GHz)", size="medium")

    wl_set = set()
    for (_, AFF, WL, _, _), df in df.groupby(  # pyright: ignore
        ["node", "affinity", "wl", "cores", "run#"]
    ):
        wl_set.add(WL)
        args = dict(
            linewidth=1.3,
            linestyle=(
                "dashed"
                if AFF == "unbalanced"
                else PLOTTING_DATA["cores"][AFF]["artist"].get_linestyle()
            ),
            color=PLOTTING_DATA["cores"][AFF]["artist"].get_color(),
            alpha=1 if AFF == "unbalanced" else 0.8,
            zorder=2 if AFF == "unbalanced" else 3,
        )
        print(args, file=sys.stderr)
        ax.plot(
            df[df.freq > 0]["at"],
            df[df.freq > 0].freq * 1e-3,
            **args,  # pyright: ignore
        )
    for wl in wl_set:
        ax = draw_compute_span(ax, wl)
    ax.set_yticks(np.arange(0, 4 + 1, 1))
    ax = set_legend(ax, legend)
    return ax


def main():
    plt.rcParams["font.family"] = "DejaVu Serif"

    df = pd.read_csv("processed/numa2.csv")
    # cherry pick run that showcases the results best
    df = df[(df["run#"] == 2)]
    # Remove first second and shift back one second, it is pointless to keep it
    df = df[(df["at"] >= 1)]
    df["at"] = df["at"] - 1
    df["in_cstate"] = df["freq"] < 1000
    print(df)
    assert isinstance(df, pd.DataFrame)

    group_cols = ["node", "affinity", "wl", "cores", "run#", "at"]
    df_pwr = (
        df.groupby(group_cols)[["duration_time", "power/energy-pkg/", "pwr"]]
        .agg(["mean", "std"])
        .reset_index(drop=False)
    )

    def cstate_count_and_avg_freq(g: pd.DataFrame) -> pd.Series:
        values = {
            "in_cstate": g["in_cstate"].astype(int).sum(),
            "corecount": g.shape[0],
            "freq": g[~g["in_cstate"]]["freq"].mean(),
        }
        result = pd.Series(values)
        result = result.fillna(0)
        return result

    df_freq = (
        df.groupby(group_cols)[["freq", "in_cstate"]]
        .apply(cstate_count_and_avg_freq)
        .reset_index(drop=False)
    )
    print(df_freq)

    assert isinstance(df_pwr, pd.DataFrame)
    fig, ax = plt.subplots(
        nrows=1, ncols=3, sharex=True, figsize=(10, 2.25), tight_layout=True
    )
    ax[0] = plot_pwr(df_pwr, ax[0], legend="cores")
    ax[1] = plot_cs(df_freq, ax[1], legend=None)
    ax[2] = plot_freq(df_freq, ax[2], legend=None)
    ax[1].set_xlabel("Time (s)")
    with pdf.PdfPages("figure/plot_numa2.pdf") as f:
        f.savefig(fig)


if __name__ == "__main__":
    main()
