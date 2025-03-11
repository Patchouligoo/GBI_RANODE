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
    TranvalSplitRandomMixin,
    TemplateRandomMixin,
    TranvalSplitUncertaintyMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    TestSetMixin,
    BkgModelMixin,
    WScanMixin,
)
from src.tasks.preprocessing import PreprocessingTrainval, PreprocessingTest
from src.tasks.bkgtemplate import PredictBkgProbTrainVal, PredictBkgProbTest
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

    scan_index = luigi.ListParameter(default=[0, 3, 5, 6, 7])

    def requires(self):
        # return [
        #     FittingScanResults.req(self, s_ratio_index=index)
        #     for index in self.scan_index
        # ]
        return [
            FittingScanResults.req(self, s_ratio_index=0),
            FittingScanResults.req(self, s_ratio_index=3),
            FittingScanResults.req(self, s_ratio_index=5),
            FittingScanResults.req(self, s_ratio_index=6, w_min=0.0001),
            FittingScanResults.req(self, s_ratio_index=7),
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


class FittingValResults(
    SigTemplateTrainingUncertaintyMixin,
    TranvalSplitUncertaintyMixin,
    WScanMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    def requires(self):

        trainval_seed_results = {}

        for index in range(self.split_num_sig_templates):
            trainval_seed_results[f"trainval_seed_{index}"] = (
                CoarseScanRANODEFixedSplitSeed.req(self, trainval_split_seed=index)
            )

        return trainval_seed_results

    def output(self):
        return {
            "scan_result": self.local_target("scan_result.json"),
            "scan_plot": self.local_target("scan_plot.pdf"),
        }

    @law.decorator.safe_output
    def run(self):

        w_range = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )
        w_range_log = np.log10(w_range)

        x_pred = None
        y_pred_list = []
        num_events = None
        CI95_drop = None

        # load scan results
        for trainval_split_index in range(self.split_num_sig_templates):
            scan_result = json.load(
                open(
                    self.input()[f"trainval_seed_{trainval_split_index}"][
                        "scan_result"
                    ].path,
                    "r",
                )
            )
            x_pred = scan_result["x_pred"]
            y_pred = scan_result["y_pred"]
            num_events = np.log(2) / scan_result["CI_95_likelihood_drop"]
            CI95_drop = scan_result["CI_95_likelihood_drop"]

            y_pred_list.append(y_pred)

        # shift all likelihoods y max to the same value
        for i in range(len(y_pred_list)):
            y_pred_list[i] = y_pred_list[i] - np.max(y_pred_list[i])

        y_pred_list = np.array(y_pred_list)
        y_mean = np.mean(y_pred_list, axis=0)
        y_std = np.std(y_pred_list, axis=0)
        y_upper = y_mean + 1.96 * y_std
        y_lower = y_mean - 1.96 * y_std

        max_likelihood_index = np.argmax(y_mean)
        max_likelihood = y_mean[max_likelihood_index]
        log_mu_pred = x_pred[max_likelihood_index]
        mu_pred = 10**log_mu_pred
        CI95_value = max_likelihood - CI95_drop

        # find left and right CI95 crossing points
        # using upper bound to be more conservative
        diff = y_upper - CI95_value
        from src.utils.utils import find_zero_crossings

        # Get all zero-crossing points
        crossings = find_zero_crossings(x_pred, diff)
        # Separate them into those on the left vs. right of the maximum
        left_crossings = [c for c in crossings if c < log_mu_pred]
        right_crossings = [c for c in crossings if c > log_mu_pred]
        # If there is more than one intersection on each side, we only want
        # the first one to the left and the first one to the right.
        x_left = max(left_crossings) if left_crossings else None
        x_right = min(right_crossings) if right_crossings else None

        mu_left = 10**x_left if x_left is not None else 0
        mu_right = 10**x_right if x_right is not None else 1

        self.output()["scan_result"].parent.touch()
        f = plt.figure()

        plt.plot(x_pred, y_mean, label="mean log_likelihood", color="red")
        plt.fill_between(
            x_pred, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, alpha=0.2, color="red"
        )

        plt.scatter(
            [log_mu_pred],
            [max_likelihood],
            color="red",
            label=f"pred mu = {mu_pred:.4f}",
        )

        plt.axvline(
            np.log10(self.s_ratio), color="black", linestyle="--", label="true w"
        )
        plt.axhline(
            CI95_value,
            color="blue",
            linestyle=":",
            label=f"CI95, [{mu_left:.6f}, {mu_right:.6f}]",
        )
        plt.xlabel("log10(w)")
        plt.ylabel("relative likelihood")
        plt.legend()
        plt.savefig(self.output()["scan_plot"].path)
        plt.tight_layout()
        plt.close(f)


# class PerformanceEvaluation(
#     CoarseScanRANODEoverW,
# ):

#     num_fine_scan = luigi.IntParameter(default=10)
#     device = luigi.Parameter(default="cuda")

#     def requires(self):

#         model_list = {}
#         w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

#         for fine_scan_index in range(self.num_fine_scan):
#             model_list[f"model_{fine_scan_index}"] = [RNodeTemplate.req(self, w_value=w_range[fine_scan_index], train_random_seed=(i)) for i in range(self.num_sig_templates)]

#         return {
#             "models": model_list,
#             "scan_result": CoarseScanRANODEoverW.req(self),
#             "test_data": Preprocessing.req(self),
#             "bkgprob": PredictBkgProb.req(self),
#         }

#     def output(self):
#         return {
#             "performance_plot": self.local_target("performance_plot.pdf"),
#             "sic_values": self.local_target("sic_values.json"),
#         }

#     @law.decorator.safe_output
#     def run(self):

#         # load the best models
#         with open(self.input()["scan_result"]["peak_info"].path, 'r') as f:
#             scan_result = json.load(f)

#         best_model_index = scan_result["mu_best_index"]

#         model_best_list = []
#         model_loss_list = []

#         for rand_seed_index in range(self.num_sig_templates):
#             model_best_seed_i = self.input()["models"][f"model_{best_model_index}"][rand_seed_index]["sig_models"]
#             metadata_best_seed_i = self.input()["models"][f"model_{best_model_index}"][rand_seed_index]["metadata"].load()

#             for model in model_best_seed_i:
#                 model_best_list.append(model.path)
#             model_loss_list.extend(metadata_best_seed_i["min_val_loss_list"])

#         # select 20 best models
#         # model_loss_list = np.array(model_loss_list)
#         # model_loss_list = np.sort(model_loss_list)[:20]
#         # model_best_list = model_best_list[:20]

#         # load test data
#         data_test_SR_model_S = np.load(self.input()['test_data']['data_test_SR_model_S'].path)

#         # load bkg prob
#         data_test_SR_prob_B = np.load(self.input()['bkgprob']['log_B_test'].path)
#         data_test_SR_prob_B = np.exp(data_test_SR_prob_B.flatten())

#         from src.models.train_model_S import pred_model_S

#         prob_S_list = []

#         for model_index, model_dir in enumerate(model_best_list):
#             print(f"evaluating model {model_index}")
#             prob_S = pred_model_S(model_dir, data_test_SR_model_S, batch_size=2048, device=self.device)
#             prob_S_list.append(prob_S)

#         prob_S_list = np.array(prob_S_list)
#         prob_S_list = np.mean(prob_S_list, axis=0)

#         prob_anomaly = prob_S_list / (1e-10 + data_test_SR_prob_B) + 1e-10
#         prob_anomaly = np.log(prob_anomaly)
#         # adjust to 0-1
#         prob_anomaly = (prob_anomaly - prob_anomaly.min()) / (prob_anomaly.max() - prob_anomaly.min())

#         truth_label = data_test_SR_model_S[:, -1].flatten()

#         from sklearn.metrics import roc_curve, roc_auc_score

#         fpr, tpr, _ = roc_curve(truth_label, prob_anomaly)
#         sic = tpr / np.sqrt(fpr)

#         # fine sic curve at fpr = 0.001
#         arg_fpr = np.argmin(np.abs(fpr - 0.001))
#         sic_value = sic[arg_fpr]

#         self.output()["performance_plot"].parent.touch()

#         with PdfPages(self.output()["performance_plot"].path) as pdf:
#             bins=np.linspace(0, 1, 100)
#             f = plt.figure()
#             plt.hist(prob_anomaly[truth_label == 0], bins=bins, label='bkg', density=True, histtype='step', lw=3)
#             plt.hist(prob_anomaly[truth_label == 1], bins=bins, label='sig', density=True, histtype='step', lw=3)
#             plt.xlabel('anomaly score')
#             plt.ylabel('num events')
#             plt.title('Anomaly score distribution')
#             plt.legend()
#             plt.yscale('log')
#             pdf.savefig(f)
#             plt.close(f)

#             f = plt.figure()
#             plt.plot(tpr, sic, label='SIC')
#             plt.xlabel('TPR')
#             plt.ylabel('SIC')
#             plt.title('SIC vs TPR')
#             plt.legend()
#             pdf.savefig(f)
#             plt.close(f)

#         with open(self.output()["sic_values"].path, 'w') as f:
#             json.dump({"sic_value_fpr001": sic_value}, f, cls=NumpyEncoder)


# # class ScanOverTruthMu(
# #     FineScanRANODEoverW,
# # ):

# #     def requires(self):
# #         truth_mu_scan_list = [0.0005, 0.001, 0.005, 0.01]

# #         return {
# #             "fine_scan_result": [FineScanRANODEoverW.req(self, s_ratio=truth_mu) for truth_mu in truth_mu_scan_list],
# #             "sic_result": [PerformanceEvaluation.req(self, s_ratio=truth_mu) for truth_mu in truth_mu_scan_list],
# #         }

# #     def output(self):
# #         return self.local_target("scan_plot.pdf")

# #     @law.decorator.safe_output
# #     def run(self):

# #         truth_mu_scan_list = [0.0005, 0.001, 0.005, 0.01]

# #         mu_pred_list = []
# #         mu_lowerbound_list = []
# #         mu_upperbound_list = []
# #         sic_values = []

# #         for index, truth_mu in enumerate(truth_mu_scan_list):

# #             fine_scan_result = json.load(open(self.input()["fine_scan_result"][index]["scan_result"].path, 'r'))
# #             sic_value = json.load(open(self.input()["sic_result"][index]["sic_values"].path, 'r'))["sic_value_fpr001"]

# #             mu_pred_i = fine_scan_result["mu_pred"]
# #             mu_lowerbound_i = fine_scan_result["mu_lowerbound"]
# #             mu_upperbound_i = fine_scan_result["mu_upperbound"]

# #             mu_pred_list.append(mu_pred_i)
# #             mu_lowerbound_list.append(mu_lowerbound_i)
# #             mu_upperbound_list.append(mu_upperbound_i)
# #             sic_values.append(sic_value)

# #         # plot
# #         with PdfPages(self.output().path) as pdf:
# #             f = plt.figure()
# #             plt.plot(truth_mu_scan_list, mu_pred_list, label='pred $\mu$', color='red')
# #             plt.fill_between(truth_mu_scan_list, mu_lowerbound_list, mu_upperbound_list, alpha=0.2, color='red')
# #             plt.plot(truth_mu_scan_list, truth_mu_scan_list, label='true $\mu$', color='black')
# #             plt.xlabel('true $\mu$')
# #             plt.ylabel('pred $\mu$')
# #             plt.xscale('log')
# #             plt.yscale('log')
# #             plt.title('Scan over truth $\mu$')
# #             plt.legend()
# #             pdf.savefig(f)
# #             plt.close(f)

# #             f = plt.figure()
# #             plt.plot(truth_mu_scan_list, sic_values, label='SIC', color='red')
# #             plt.xlabel('true $\mu$')
# #             plt.ylabel('SIC')
# #             plt.xscale('log')
# #             plt.title('SIC vs true $\mu$')
# #             plt.legend()
# #             pdf.savefig(f)
# #             plt.close(f)
