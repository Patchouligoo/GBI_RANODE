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
    CoarseScanRANODEoverW,
    RNodeTemplate,
    CoarseScanRANODEFixedSplitSeed,
)


class FittingScanResults(
    CoarseScanRANODEoverW,
):

    def requires(self):
        return CoarseScanRANODEoverW.req(self)

    def output(self):
        return {
            "scan_plot": self.local_target("scan_plot.pdf"),
            "peak_info": self.local_target("peak_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        # load scan results
        prob_S_scan = np.load(self.input()["prob_S_scan"].path)
        prob_B_scan = np.load(self.input()["prob_B_scan"].path)
        w_scan_range = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )
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
        return self.local_target("scan_plot.pdf")

    @law.decorator.safe_output
    def run(self):

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

            plt.title("Scan over true $\mu$")
            plt.legend()
            pdf.savefig(f)
            plt.close(f)
