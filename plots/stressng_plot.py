#!/usr/bin/env python3

from typing import Literal
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

from matplotlib.axes import Axes
from matplotlib import colormaps as CM
from matplotlib.patches import Rectangle

pastel2 = CM.get_cmap("Pastel2")
set2 = CM.get_cmap("Set2")
tab20b = CM.get_cmap("tab20b")
tab20c = CM.get_cmap("tab20c")

PLOTTING_DATA = {
    "wl": {
        "cpu": {"line": "dashdot", "color": set2(1), "color2": tab20c(3)},
        "stream": {"line": "dashed", "color": set2(2), "color2": tab20c(7)},
    },
    "mcnode26": {
        "marker": {"shape": "o", "size": 3},
    },
    "mcnode44": {
        "marker": {"shape": "^", "size": 3},
    },
}

ARTISTS_NODE = [
    Line2D(
        [0],
        [0],
        linestyle="none",
        color="black",
        marker=PLOTTING_DATA["mcnode26"]["marker"]["shape"],
        markersize=PLOTTING_DATA["mcnode26"]["marker"]["size"],
    ),
    Line2D(
        [0],
        [0],
        linestyle="none",
        color="black",
        marker=PLOTTING_DATA["mcnode44"]["marker"]["shape"],
        markersize=PLOTTING_DATA["mcnode44"]["marker"]["size"],
    ),
]

ARTISTS_WL = [
    Rectangle((0, 0), 0, 0, color=PLOTTING_DATA["wl"]["cpu"]["color"]),
    Rectangle((0, 0), 0, 0, color=PLOTTING_DATA["wl"]["stream"]["color"]),
]


def plot_pwr(df1: pd.DataFrame, ax: Axes, legend: Literal["node", "workload"]) -> Axes:
    ax.set_title("\u0394 Power Cons. (W)", size="medium")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# stressors (cores)")
    ax.grid(axis="y", linestyle="dotted")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v))))

    for (nd, wl), df in df1.groupby(level=(0, 1)):  # pyright: ignore
        df = df.droplevel((0, 1))
        args = dict(
            linewidth=1,
            linestyle=PLOTTING_DATA["wl"][wl]["line"],
            marker=PLOTTING_DATA[nd]["marker"]["shape"],
            markersize=PLOTTING_DATA[nd]["marker"]["size"],
            color=PLOTTING_DATA["wl"][wl]["color"],
        )
        ax.errorbar(
            x=df.cores_used,
            y=df.delta_pwr["mean"],
            yerr=df.delta_pwr["std"],
            **args,  # pyright: ignore
        )
        ax.set_xticks(df.cores_used.tolist())

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yticks(np.arange(ax.get_yticks()[0], ax.get_yticks()[-1], 50))

    lgnd_args = dict(framealpha=0.8, ncol=1, handlelength=0.8, handleheight=0.8, markerscale=1.6)
    if legend == "node":
        ax.legend(ARTISTS_NODE, ["mcnode26", "mcnode44"], loc="upper left", **lgnd_args)
    elif legend == "workload":
        ax.legend(ARTISTS_WL, ["--cpu", "--stream"], loc="upper right", **lgnd_args)
    else:
        assert False
    return ax


def plot_delta(df: pd.DataFrame, ax: Axes, legend: Literal["node", "workload"]) -> Axes:
    ax.set_title("Inc. Power/core (W)", size="medium")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# stressors (cores)")
    ax.grid(axis="y", linestyle="dotted")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v))))

    for (nd, wl), df in df.groupby(level=(0, 1)):  # pyright: ignore
        df = df.droplevel((0, 1))
        args = dict(
            linewidth=1,
            linestyle=PLOTTING_DATA["wl"][wl]["line"],
            marker=PLOTTING_DATA[nd]["marker"]["shape"],
            markersize=PLOTTING_DATA[nd]["marker"]["size"],
            color=PLOTTING_DATA["wl"][wl]["color"],
        )
        ax.plot(
            df.cores_used[1:],
            df.inc_pwr_scaled["mean"][1:],
            **args,  # pyright: ignore
        )
        ax.set_xticks(df.cores_used[1:].tolist())

    max_tick = ax.get_yticks(minor=False)[-1]
    ax.set_yticks(np.arange(0, max_tick, 1))
    ax.set_ylim(-0.5, ax.get_ylim()[1])

    lgnd_args = dict(framealpha=0.8, ncol=1, handlelength=0.8, handleheight=0.8)
    if legend == "node":
        ax.legend(ARTISTS_NODE, ["mcnode26", "mcnode44"], loc="upper left", **lgnd_args)
    elif legend == "workload":
        ax.legend(ARTISTS_WL, ["--cpu", "--stream"], loc="upper right", **lgnd_args)
    else:
        assert False
    return ax


def main():
    plt.rcParams["font.family"] = "DejaVu Serif"

    df_idle = pd.read_csv("processed/idle.csv")
    df_idle = df_idle[["node", "cstates", "latency", "setfreq", "pwr"]]
    df_idle.rename({"pwr": "idle_pwr"}, axis=1, inplace=True)

    df = pd.read_csv("processed/stressng.csv")
    df = pd.merge(df, df_idle, on=("node", "cstates", "latency", "setfreq"))
    df["delta_pwr"] = df["pwr"] - df["idle_pwr"]

    def create_inc_pwr(g: pd.DataFrame) -> pd.DataFrame:
        delta_col = ("delta_pwr", "mean")
        inc_col = ("inc_pwr", "mean")
        scaled_col = ("inc_pwr_scaled", "mean")
        g[inc_col] = g[delta_col] - g[delta_col].shift(1)
        g[inc_col] = g[inc_col].fillna(0)
        g[scaled_col] = g[inc_col] / g["cores_used"]
        return g

    df_cstates = (
        df[df["cstates"] == True]
        .groupby(["node", "workload", "cores_used"])[["pwr", "freq", "delta_pwr"]]
        .agg(["mean", "std"])
        .reset_index(drop=False)
    )
    df_cstates = df_cstates.groupby(["node", "workload"]).apply(
        create_inc_pwr, include_groups=False
    )
    df_nocstates = (
        df[(df["cstates"] == False) & (df["latency"] == 0)]
        .groupby(["node", "setfreq", "workload", "cores_used"])[
            ["pwr", "freq", "delta_pwr"]
        ]
        .agg(["mean", "std"])
    )
    assert isinstance(df_cstates, pd.DataFrame)
    assert isinstance(df_nocstates, pd.DataFrame)

    fig, ax = plt.subplots(ncols=2, figsize=(5, 2.8), tight_layout=True)
    ax[0] = plot_pwr(df_cstates, ax[0], legend="node")
    ax[1] = plot_delta(df_cstates, ax[1], legend="workload")
    with pdf.PdfPages("figure/plot_stressng.pdf") as f:
        f.savefig(fig)


if __name__ == "__main__":
    main()
