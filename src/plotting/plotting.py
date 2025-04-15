import numpy as np
import pandas as pd
import os
from quickstats.plots import General1DPlot, TwoPanel1DPlot
import matplotlib.pyplot as plt
from functools import partial
from quickstats.plots import DataModelingPlot
from quickstats.plots.colors import get_cmap, get_rgba
from quickstats.concepts import Histogram1D
from quickstats.maths.histograms import bin_center_to_bin_edge


def mu2sig(mu, B: int):
    # change this to the actual number of background events used in training + validatio
    S = B * mu
    return S / B**0.5


def sig2mu(sig, B):
    # change this to the actual number of background events used in training + validation
    S = sig * B**0.5
    return S / B


def plot_mu_scan_results(
    dfs,
    metadata,
    output_path,
):

    # -------------------- plotting settings --------------------
    colors = get_cmap("simple_contrast").colors
    styles = {
        "plot": {"marker": "o"},
        "legend": {"fontsize": 15},
        "ratio_frame": {"height_ratios": (2, 1), "hspace": 0.05},
    }
    styles_map = {
        "true": {
            "plot": {"color": "hdbs:spacecadet"},
            "fill_between": {"color": "none"},
        },
        "predicted": {
            "plot": {"color": "hdbs:pictorialcarmine"},
            "fill_between": {
                "facecolor": get_rgba(colors[0], 0.2),
                "alpha": None,
                "edgecolor": get_rgba(colors[0], 0.9),
            },
        },
    }
    config = {
        "error_on_top": False,
        "inherit_color": False,
        # 'draw_legend': False,
    }
    label_map = {
        "true": "Truth",
        "predicted": "Predicted",
    }

    # -------------------- making plots --------------------
    mx = metadata["mx"]
    my = metadata["my"]
    num_B = metadata["num_B"]
    use_full_stats = metadata["use_full_stats"]
    use_perfect_bkg_model = metadata["use_perfect_modelB"]
    use_bkg_model_gen_data = metadata["use_modelB_genData"]

    if use_full_stats:
        text = f"Full stats"
    else:
        text = f"Lumi matched"
    if use_perfect_bkg_model:
        text += ", model B trained in SR"
    if use_bkg_model_gen_data:
        text += ", use model B to generate bkgs in data"

    text += f"//Signal at $(m_X, m_Y) = ({mx}, {my})$ GeV"

    xticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
    xticklabels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "5"]
    xticks2 = mu2sig(np.array(xticks), B=num_B)
    xticklabels2 = [str(round(v, 2)) for v in xticks2]
    xlabel = "Signal Injection (%)"
    plotter = General1DPlot(
        dfs, styles=styles, styles_map=styles_map, label_map=label_map, config=config
    )
    plotter.add_text(text, 0.05, 0.95, fontsize=18)

    ax = plotter.draw(
        "x",
        "y",
        targets=["true", "predicted"],
        yerrloattrib="yerrlo",
        yerrhiattrib="yerrhi",
        xlabel=xlabel,
        ylabel=r"$\mu\,(\%)$",
        ymin=2e-5,
        ymax=0.1,
        logx=True,
        logy=True,
        offset_error=False,
        legend_order=["true", "predicted"],
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    mu2sig_f = partial(mu2sig, B=num_B)
    sig2mu_f = partial(sig2mu, B=num_B)
    ax2 = ax.secondary_xaxis("top", functions=(mu2sig_f, sig2mu_f))
    ax2.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.set_xticks(xticks2)
    ax2.set_xticklabels(xticklabels2)
    ax2.set_xlabel(r"$S/\sqrt{B}$", labelpad=10, fontsize=18)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mu_scan_results_multimodels(
    dfs,
    metadata,
    output_path,
):

    # -------------------- plotting settings --------------------
    colors = get_cmap("simple_contrast").colors
    styles = {
        "plot": {"marker": "o"},
        "legend": {"fontsize": 15},
        "ratio_frame": {"height_ratios": (2, 1), "hspace": 0.05},
    }
    styles_map = {
        "true": {
            "plot": {"color": "hdbs:spacecadet"},
            "fill_between": {"color": "none"},
        },
    }

    key_list = [key for key in dfs.keys() if key != "true"]

    for index, key in enumerate(key_list):
        styles_map[key] = {
            "plot": {"color": colors[index]},
            "fill_between": {
                "facecolor": get_rgba(colors[index], 0.2),
                "alpha": None,
                "edgecolor": get_rgba(colors[index], 0.9),
            },
        }

    config = {
        "error_on_top": False,
        "inherit_color": False,
        # 'draw_legend': False,
    }
    label_map = {
        "true": "Truth",
    }
    for index, key in enumerate(key_list):
        label_map[key] = f"Predicted {key}"

    # -------------------- making plots --------------------
    mx = metadata["mx"]
    my = metadata["my"]
    num_B = metadata["num_B"]
    use_full_stats = metadata["use_full_stats"]

    if use_full_stats:
        text = f"Full stats"
    else:
        text = f"Lumi matched"

    text += f"//Signal at $(m_X, m_Y) = ({mx}, {my})$ GeV"

    xticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
    xticklabels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "5"]
    xticks2 = mu2sig(np.array(xticks), B=num_B)
    xticklabels2 = [str(round(v, 2)) for v in xticks2]
    xlabel = "Signal Injection (%)"
    plotter = General1DPlot(
        dfs, styles=styles, styles_map=styles_map, label_map=label_map, config=config
    )
    plotter.add_text(text, 0.05, 0.95, fontsize=18)

    ax = plotter.draw(
        "x",
        "y",
        targets=dfs.keys(),
        yerrloattrib="yerrlo",
        yerrhiattrib="yerrhi",
        xlabel=xlabel,
        ylabel=r"$\mu\,(\%)$",
        ymin=2e-5,
        ymax=0.1,
        logx=True,
        logy=True,
        offset_error=False,
        # legend_order=["true", "predicted"],
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    mu2sig_f = partial(mu2sig, B=num_B)
    sig2mu_f = partial(sig2mu, B=num_B)
    ax2 = ax.secondary_xaxis("top", functions=(mu2sig_f, sig2mu_f))
    ax2.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.set_xticks(xticks2)
    ax2.set_xticklabels(xticklabels2)
    ax2.set_xlabel(r"$S/\sqrt{B}$", labelpad=10, fontsize=18)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
