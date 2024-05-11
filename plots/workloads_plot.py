#!/usr/bin/env python3

import copy
from typing import List, Tuple
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

from matplotlib.axes import Axes
from matplotlib import colormaps as CM

cmap = CM.get_cmap("Dark2")


def plot_pwr(
    df: pd.DataFrame, ax_pwr: Axes, ax_t: Axes, ax_e: Axes
) -> Tuple[Axes, Axes, Axes]:
    import sys

    ax_pwr.grid(visible=True, axis="y", linestyle="dotted")
    ax_pwr.set_title("\u0394 Power (W)", size="medium")

    ax_t.grid(visible=True, axis="y", linestyle="dotted")
    ax_t.set_title("Time, scaled vs. time at LNLP", size="medium")
    ax_t.set_xlabel("Processor Frequency (GHz)")

    ax_e.grid(visible=True, axis="y", linestyle="dotted")
    ax_e.set_title("\u0394 energy, scaled vs. energy at LNLP", size="medium")

    PLOTTING_DATA = {
        "mcnode26": {
            "artist": Line2D([], [], linewidth=0, color="black", marker="o"),
            "marker": {
                "shape": "o",
                "size": 4,
            },
        },
        "mcnode44": {
            "artist": Line2D([], [], linewidth=0, color="black", marker="^"),
            "marker": {
                "shape": "^",
                "size": 4,
            },
        },
        "wl": {
            "boxblur": "BB",
            "condtextgen": "TG",
            "ffmpeg": "VT",
            "imgclass": "IC",
            "matmul": "MM",
            "objdetect": "OD",
        },
    }

    artists_wl: List[Line2D] = []
    for i, ((nd, wl), df) in enumerate(  # pyright: ignore
        df.groupby(["node", "workload"])
    ):  # pyright: ignore
        scaled_freq = df["setfreq"] * 1e-3
        args = dict(
            linewidth=1.25,
            linestyle="dotted",
            marker=PLOTTING_DATA[nd]["marker"]["shape"],
            markersize=PLOTTING_DATA[nd]["marker"]["size"],
            color=cmap(i % 6),
        )
        print(args, file=sys.stderr)
        ln = ax_pwr.plot(
            scaled_freq, df["delta"]["mean"], label=wl, **args  # pyright: ignore
        )

        if i < 6:
            artists_wl.append(copy.deepcopy(ln[0]))
            artists_wl[-1].set_marker("none")

        ax_t.plot(scaled_freq, df["T_scaled"]["mean"], **args)  # pyright: ignore
        ax_e.plot(scaled_freq, df["E_scaled"]["mean"], **args)  # pyright: ignore

    top_ylim = max(ax_e.get_ylim()[1], ax_t.get_ylim()[1])
    ax_e.set_ylim(0.4, top_ylim * 1.02)
    ax_t.set_ylim(0.4, top_ylim * 1.02)

    ax_pwr.legend(
        artists_wl,
        [PLOTTING_DATA["wl"][str(x.get_label())] for x in artists_wl],
        loc="upper left",
        handlelength=1.2,
        ncol=2,
    )
    ax_t.legend(
        [v["artist"] for k, v in PLOTTING_DATA.items() if k.startswith("mcnode")],
        (x for x in PLOTTING_DATA.keys() if x.startswith("mcnode")),
        loc="upper right",
        ncol=1,
    )
    return ax_pwr, ax_t, ax_e


def main():
    plt.rcParams["font.family"] = "DejaVu Serif"

    df_i = pd.read_csv("processed/idle.csv")
    df_i = df_i[["node", "cstates", "latency", "setfreq", "pwr"]]
    df_i = df_i[
        (df_i["cstates"] == True)
        | ((df_i["cstates"] == False) & (df_i["latency"] == 0))
    ]
    assert isinstance(df_i, pd.DataFrame)
    df_i.drop("latency", axis=1, inplace=True)
    df_i.rename({"pwr": "idle_pwr"}, axis=1, inplace=True)
    df = pd.read_csv("processed/workloads.csv")

    agg_cols = [
        # "cache-misses",
        # "cache-references",
        "duration_time",
        "cycles",
        "instructions",
        "pwr",
        "freq",
        "delta",
        "deltanrg",
    ]

    df = pd.concat(
        [
            pd.merge(
                df[df["cstates"] == True],
                df_i[df_i["cstates"] == True]  # pyright: ignore
                .groupby(["node", "cstates"])["idle_pwr"]
                .agg(["mean"]),
                on=["node", "cstates"],
            ),
            pd.merge(
                df[df["cstates"] == False],
                df_i[df_i["cstates"] == False]  # pyright: ignore
                .groupby(["node", "cstates", "setfreq"])["idle_pwr"]
                .agg(["mean"]),
                on=["node", "cstates", "setfreq"],
            ),
        ]
    )
    df.rename({"mean": "delta"}, axis=1, inplace=True)
    df["delta"] = df["pwr"] - df["delta"]
    df["deltanrg"] = df["delta"] * (df["duration_time"] * 1e-9)
    df = (
        df.groupby(["node", "cstates", "workload", "setfreq"])[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    df = df[df["cstates"] == True]
    df.reset_index(drop=True)

    def divide_by_first_row(group):
        for x, y in zip(("duration_time", "deltanrg"), ("T_scaled", "E_scaled")):
            first_row_value = group.iloc[0][x]
            div = group[x] / first_row_value["mean"]
            group[(y, "mean")] = div["mean"]
            group[(y, "std")] = div["std"]
        return group

    df = df.groupby(["node", "cstates", "workload"]).apply(divide_by_first_row)
    df.drop(["node", "cstates", "workload"], axis=1, inplace=True)
    df.to_csv("processed/workloads_tmp.csv")

    fig, ax = plt.subplots(
        nrows=1, ncols=3, sharex=True, figsize=(10, 2.5), tight_layout=True
    )
    ax[0], ax[1], ax[2] = plot_pwr(df, ax[0], ax[1], ax[2])
    with pdf.PdfPages("figure/plot_workloads.pdf") as f:
        f.savefig(fig)


if __name__ == "__main__":
    main()
