#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import copy

from matplotlib.axes import Axes
from matplotlib import colormaps as CM
from matplotlib.patches import Rectangle

pastel2 = CM.get_cmap("Pastel2")
set2 = CM.get_cmap("Set2")
tab20b = CM.get_cmap("tab20b")
tab20c = CM.get_cmap("tab20c")


def plot_pwr(df_cstates: pd.DataFrame, df_nocstates: pd.DataFrame, ax: Axes) -> Axes:
    import sys

    bar_width = 0.10
    tick_width = 0.5

    ax.set_xlim(0.8, 2.8)
    ax.set_xticks(
        [1 + bar_width / 2 + tick_width * i for i in range(4)],
        ["C-States On", "LNLP", "Nom.", "Max"],
    )
    ax.grid(visible=True, axis="y", linestyle="dotted")
    ax.set_title("Idle Power (W)", size="medium")
    ax.set_xlabel("Processor Frequency")

    bc1 = ax.bar(
        x=[1 + bar_width * i for i in range(2)],
        width=bar_width,
        height=df_cstates["pwr"]["mean"],
        color=pastel2(1),
        hatch=["...", "xxx"],
        edgecolor=set2(1),
        align="center",
        linewidth=1,
    )
    ax.errorbar(
        [1 + bar_width * i for i in range(2)],
        df_cstates["pwr"]["mean"],
        yerr=df_cstates["pwr"]["std"],
        fmt="none",
        color="black",
    )
    for v, rect in zip(bc1.datavalues, bc1.patches):  # pyright: ignore
        rect: Rectangle
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 5,
            str(round(v, 2)),
            ha="center",
            rotation=90,
            fontsize=10,
        )
    ax.axvline(1.25, linestyle="solid", color="black", linewidth="1")

    PLOTTING_DATA = {
        "on": {},
        "latency": {
            0: {
                "color": tab20b(15),
                "edgecolor": tab20b(13),
                "offset": 5,
                "artist": None,
            },
            1: {
                "color": tab20b(11),
                "edgecolor": tab20b(9),
                "offset": -5,
                "artist": None,
            },
            30: {
                "color": tab20b(7),
                "edgecolor": tab20b(5),
                "offset": -5,
                "artist": None,
            },
            800: {
                "color": tab20b(7),
                "edgecolor": tab20b(5),
                "offset": -5,
                "artist": None,
            },
        },
        "mcnode26": {
            "hatch": "...",
            "coords": [1 + tick_width * i for i in range(1, 4)],
            "artist": None,
        },
        "mcnode44": {
            "hatch": "xxx",
            "coords": [1 + bar_width + tick_width * i for i in range(1, 4)],
            "artist": None,
        },
    }

    for l0, y0 in df_nocstates.groupby(level=0):  # pyright: ignore
        y0 = y0.droplevel(0)
        for l1, y1 in y0.groupby(level=0):  # pyright: ignore
            y1 = y1.droplevel(0)
            l0: str
            l1: str
            args = dict(
                x=PLOTTING_DATA[l0]["coords"],
                width=bar_width,
                height=y1["pwr"]["mean"],
                color=PLOTTING_DATA["latency"][l1]["color"],
                hatch=PLOTTING_DATA[l0]["hatch"],
                edgecolor=PLOTTING_DATA["latency"][l1]["edgecolor"],
                align="center",
                linewidth=1,
            )
            print(args, file=sys.stderr)
            bc = ax.bar(**args)  # pyright: ignore
            ax.errorbar(
                PLOTTING_DATA[l0]["coords"],
                y1["pwr"]["mean"],
                yerr=y1["pwr"]["std"],
                fmt="none",
                color="black",
            )
            artist = copy.deepcopy(bc.patches[0])
            artist.set_hatch("")
            PLOTTING_DATA["latency"][l1]["artist"] = artist
    ax.set_ylim(0, ax.get_ylim()[1] + 10)
    ax.set_yticks(np.arange(0, ax.get_yticks(minor=False)[-1] + 1, 25))

    patches_cpy = [copy.deepcopy(x) for x in bc1.patches]
    for x in patches_cpy:
        x.set_facecolor("none")
        x.set_edgecolor("none")

    lgnd = ax.legend(
        patches_cpy
        + [PLOTTING_DATA["latency"][x]["artist"] for x in (0, 1, 30)]
        + [copy.deepcopy(bc1.patches[0])],
        ["mcnode26", "mcnode44"] + ["CS Off", "C2, C1 Off", "C2 Off", "CS On"],
        framealpha=1,
        loc="upper left",
        ncol=2,
        fontsize=9,
        handlelength=0.8,
        handleheight=0.8,
    )
    lgnd.legend_handles[-1].set_hatch("")  # pyright: ignore
    return ax


def plot_freq(df1: pd.DataFrame, df2: pd.DataFrame, ax: Axes) -> Axes:
    import sys

    ax.set_title("Observed Frequency (MHz)", size="medium")
    ax.set_xlabel("Configured Frequency (MHz)")

    PLOTTING_DATA = {
        "latency": {
            0: {
                "marker": {
                    "color": tab20b(15),
                    "edgecolor": tab20b(13),
                },
            },
            1: {
                "marker": {
                    "color": tab20b(11),
                    "edgecolor": tab20b(10),
                },
            },
            30: {
                "marker": {
                    "color": tab20b(7),
                    "edgecolor": tab20b(6),
                },
            },
            800: {
                "marker": {
                    "color": tab20b(7),
                    "edgecolor": tab20b(6),
                },
            },
        },
        "mcnode26": {
            "maxfreq": None,
            "artist": None,
            "marker": {
                "shape": "o",
                "color": tab20c(20),
                "edgecolor": "black",
                "size": 6,
            },
        },
        "mcnode44": {
            "maxfreq": None,
            "artist": None,
            "marker": {
                "shape": "^",
                "color": tab20c(20),
                "edgecolor": "black",
                "size": 6,
            },
        },
    }

    artists_freq = []
    artists_node = []
    xticks = []

    # Plot the expected values
    first_iter = True
    for l0, y0 in df2.groupby(level=0):  # pyright: ignore
        l0: str
        y0 = y0.droplevel(0)
        l1, y1 = next(iter(y0.groupby(level=0)))  # pyright: ignore
        y1 = y1.droplevel(0)
        args = dict(
            linestyle="solid" if first_iter else "none",
            color=PLOTTING_DATA[l0]["marker"]["edgecolor"],
            marker=PLOTTING_DATA[l0]["marker"]["shape"],
            markersize=PLOTTING_DATA[l0]["marker"]["size"] + 2,
            markerfacecolor=PLOTTING_DATA[l0]["marker"]["color"],
            markeredgecolor=PLOTTING_DATA[l0]["marker"]["edgecolor"],
            markeredgewidth=1,
        )
        print(y1.index.values, args, file=sys.stderr)
        node = ax.plot(y1.index.values, y1.index.values, **args)  # pyright: ignore
        print(node, file=sys.stderr)
        xticks.extend(y1.index.values.tolist())
        for x, st in zip(y1.index.values, ["dashed", "dotted", "dashdot"]):
            ln = ax.axvline(
                x,
                linestyle=st,
                color="black",
                alpha=0.4,
                linewidth="1",
                zorder=1,
            )
            PLOTTING_DATA[l0]["maxfreq"] = x
            if first_iter:
                artists_freq.append(ln)
        if first_iter:
            rect = Rectangle(
                (0, 0),
                10,
                10,
                facecolor=PLOTTING_DATA[l0]["marker"]["color"],
                linewidth=0,
            )
            artists_freq.append(rect)
        artists_node.append(copy.deepcopy(node[0]))
        artists_node[-1].set_linestyle("none")
        artists_node[-1].set_markerfacecolor("none")
        first_iter = False

    artists_state = [
        Rectangle((0, 0), 0, 0, facecolor=x["marker"]["color"])
        for x in PLOTTING_DATA["latency"].values()
    ][:-1] + [Rectangle((0, 0), 0, 0, facecolor=pastel2(1))]
    for l0, y0 in df2.groupby(level=0):  # pyright: ignore
        y0 = y0.droplevel(0)
        for l1, y1 in y0.groupby(level=0):  # pyright: ignore
            l1: str
            y1 = y1.droplevel(0)
            args = dict(
                linestyle="none",
                marker=PLOTTING_DATA[l0]["marker"]["shape"],
                markersize=PLOTTING_DATA[l0]["marker"]["size"],
                markerfacecolor=PLOTTING_DATA["latency"][l1]["marker"]["color"],
                markeredgecolor=PLOTTING_DATA["latency"][l1]["marker"]["edgecolor"],
                markeredgewidth=1,
            )
            print(y1.index.values, y1["freq"]["mean"], args, file=sys.stderr)
            ln = ax.plot(y1.index.values, y1["freq"]["mean"], **args)  # pyright: ignore

    for l0, y0 in df1.groupby(level=0):  # pyright: ignore
        args = dict(
            linestyle="none",
            marker=PLOTTING_DATA[l0]["marker"]["shape"],
            markersize=PLOTTING_DATA[l0]["marker"]["size"],
            markerfacecolor=pastel2(1),
            markeredgecolor=set2(1),
            markeredgewidth=1,
        )
        print(y0["freq"]["mean"], args, file=sys.stderr)
        ax.plot(
            PLOTTING_DATA[l0]["maxfreq"], y0["freq"]["mean"], **args  # pyright: ignore
        )

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1000)
    ax.set_xticks(xticks)

    lgnd_args = dict(
        framealpha=0.8,
        fontsize=9,
        handlelength=1,
        handleheight=0.8,
        markerscale=0.8,
    )
    lgnd = ax.legend(
        artists_freq + artists_node,
        ["LNLP", "Nominal", "Max", "Configured"] + ["mcnode26", "mcnode44"],
        loc="upper left",
        ncol=2,
        **lgnd_args
    )
    ax.add_artist(lgnd)
    ax.legend(
        artists_state,
        ["CS Off", "C2, C1 Off", "C2 Off", "CS On"],
        loc=(0.53, 0.15),
        ncol=2,
        **lgnd_args
    )
    return ax


def main():
    plt.rcParams["font.family"] = "DejaVu Serif"

    df = pd.read_csv("processed/idle.csv")
    df_cstates = (
        df[df["cstates"] == True].groupby("node")[["pwr", "freq"]].agg(["mean", "std"])
    )
    df_nocstates = (
        df[df["cstates"] == False]
        .groupby(["node", "latency", "setfreq"])[["pwr", "freq"]]
        .agg(["mean", "std"])
    )
    assert isinstance(df_cstates, pd.DataFrame)
    assert isinstance(df_nocstates, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    ax = plot_pwr(df_cstates, df_nocstates, ax)
    with pdf.PdfPages("figure/plot_idle_1.pdf") as f:
        f.savefig(fig)
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    ax = plot_freq(df_cstates, df_nocstates, ax)
    with pdf.PdfPages("figure/plot_idle_2.pdf") as f:
        f.savefig(fig)


if __name__ == "__main__":
    main()
