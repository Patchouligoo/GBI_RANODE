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

            mu_true = peak_info["mu_true"]
            mu_pred = peak_info["mu_pred"]
            mu_lowerbound = peak_info["left_CI"]
            mu_upperbound = peak_info["right_CI"]

            mu_true_list.append(mu_true)
            mu_pred_list.append(mu_pred)
            mu_lowerbound_list.append(mu_lowerbound)
            mu_upperbound_list.append(mu_upperbound)

        dfs = {
            "true": pd.DataFrame(
                {
                    "x": np.array(mu_true_list),
                    "y": np.array(mu_true_list),
                }
            ),
            "predicted": pd.DataFrame(
                {
                    "x": np.array(mu_true_list),
                    "y": np.array(mu_pred_list),
                    "yerrlo": np.array(mu_lowerbound_list),
                    "yerrhi": np.array(mu_upperbound_list),
                }
            ),
        }

        misc = {
            "mx": self.mx,
            "my": self.my,
            "use_full_stats": self.use_full_stats,
            "use_perfect_modelB": self.use_perfect_bkg_model,
            "use_modelB_genData": self.use_bkg_model_gen_data,
            "num_B": num_B,
        }

        self.output().parent.touch()
        output_path = self.output().path

        from src.plotting.plotting import plot_mu_scan_results

        plot_mu_scan_results(
            dfs,
            misc,
            output_path,
        )


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
