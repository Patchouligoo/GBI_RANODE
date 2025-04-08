import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    BkgModelMixin,
    WScanMixin,
)
from src.tasks.preprocessing import PreprocessingFold
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.rnodetemplate import (
    ScanRANODE,
    RNodeTemplate,
)


class FittingScanResults(
    ScanRANODE,
):

    def requires(self):
        return ScanRANODE.req(self)

    def output(self):
        return {
            "scan_plot": self.local_target(
                f"scan_plot_{str_encode_value(self.s_ratio)}.pdf"
            ),
            "peak_info": self.local_target("peak_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        # load scan results
        prob_S_scan = np.load(self.input()["prob_S_scan"].path)
        prob_B_scan = np.load(self.input()["prob_B_scan"].path)
        w_scan_range = self.w_range
        w_true = self.s_ratio

        from src.fitting.fitting import bootstrap_and_fit

        self.output()["scan_plot"].parent.touch()
        output_dir = self.output()
        bootstrap_and_fit(prob_S_scan, prob_B_scan, w_scan_range, w_true, output_dir)


class ScanOverTrueMu(
    BkgModelMixin,
    ProcessMixin,
    BaseTask,
):

    scan_index = luigi.ListParameter(
        default=[
            0,  # 0
            5,  # 0.10%
            6,  # 0.17%
            7,  # 0.30%
            8,  # 0.53%
            9,  # 0.93%
            11,  # 2.85%
            12,  # 5.01%
        ]
    )

    def requires(self):
        return [
            FittingScanResults.req(self, s_ratio_index=index)
            for index in self.scan_index
        ]

    def output(self):
        return self.local_target("full_scan.pdf")

    @law.decorator.safe_output
    def run(self):

        if self.use_full_stats:
            num_B = 738020
        else:
            num_B = 121980

        mu_true_list = []
        mu_pred_list = []
        mu_lowerbound_list = []
        mu_upperbound_list = []

        for index in range(len(self.scan_index)):
            with open(self.input()[index]["peak_info"].path, "r") as f:
                peak_info = json.load(f)

            mu_true = peak_info["mu_true"] * 100
            mu_pred = peak_info["mu_pred"] * 100
            mu_lowerbound = peak_info["left_CI"] * 100
            mu_upperbound = peak_info["right_CI"] * 100

            mu_true_list.append(mu_true)
            mu_pred_list.append(mu_pred)
            mu_lowerbound_list.append(mu_lowerbound)
            mu_upperbound_list.append(mu_upperbound)

        # plot
        self.output().parent.touch()
        with PdfPages(self.output().path) as pdf:
            f = plt.figure()
            plt.plot(mu_true_list, mu_pred_list, color="red")
            plt.scatter(mu_true_list, mu_pred_list, label="pred $\mu$", color="red")
            plt.fill_between(
                mu_true_list,
                mu_lowerbound_list,
                mu_upperbound_list,
                alpha=0.2,
                color="red",
                label="95% CI",
            )
            plt.plot(
                np.linspace(0, 5, 100),
                np.linspace(0, 5, 100),
                label="true $\mu$",
                color="black",
            )

            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(0.008, 7)
            plt.ylim(0.008, 7)
            plt.xlabel("$\mu$ (%)")
            plt.ylabel("$\mu$ (%)")

            # set x and y axis to be non-scientific
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)

            # Set custom ticks for primary x-axis (and similarly for y-axis)
            x_ticks = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 5])
            ax.set_xticks(x_ticks)
            ax.set_yticks(x_ticks)

            # Define the transformation factor and functions
            factor = 0.01 * num_B / np.sqrt(num_B)

            def forward(x):
                return x * factor

            def inverse(x):
                return x / factor

            # Create a secondary x-axis on the top using the transformation functions
            ax2 = ax.secondary_xaxis("top", functions=(forward, inverse))
            ax2.set_xscale("log")
            top_ticks = forward(x_ticks)  # i.e. x_ticks * factor
            ax2.set_xticks(top_ticks)

            bottom_minor_ticks = ax.xaxis.get_minorticklocs()
            top_minor_ticks = forward(bottom_minor_ticks)
            ax2.set_xticks(top_minor_ticks, minor=True)

            ax2.set_xlabel("$S/\\sqrt{B}$")
            ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax2.xaxis.get_major_formatter().set_scientific(False)
            ax2.xaxis.get_major_formatter().set_useOffset(False)

            plt.legend()
            pdf.savefig(f)
            plt.close(f)


class FittingScanResultsCrossFolds(
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    WScanMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):
        return [
            ScanRANODE.req(self, fold_split_seed=index)
            for index in range(self.fold_split_num)
        ]

    def output(self):
        return {
            "scan_plot": self.local_target(
                f"scan_plot_{str_encode_value(self.s_ratio)}.pdf"
            ),
            "peak_info": self.local_target("peak_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        # load scan results
        for index in range(self.fold_split_num):
            if index == 0:
                prob_S_scan = np.load(self.input()[index]["prob_S_scan"].path)
                prob_B_scan = np.load(self.input()[index]["prob_B_scan"].path)
            else:
                prob_S_scan = np.concatenate(
                    (prob_S_scan, np.load(self.input()[index]["prob_S_scan"].path)),
                    axis=-1,
                )
                prob_B_scan = np.concatenate(
                    (prob_B_scan, np.load(self.input()[index]["prob_B_scan"].path)),
                    axis=-1,
                )
        prob_S_scan = np.array(prob_S_scan)
        prob_B_scan = np.array(prob_B_scan)

        w_scan_range = self.w_range
        w_true = self.s_ratio

        from src.fitting.fitting import bootstrap_and_fit

        self.output()["scan_plot"].parent.touch()
        output_dir = self.output()
        bootstrap_and_fit(prob_S_scan, prob_B_scan, w_scan_range, w_true, output_dir)
