import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.utils.law import BaseTask, SignalStrengthMixin, TemplateRandomMixin, SigTemplateUncertaintyMixin, ProcessMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.rnodetemplate import CoarseScanRANODEoverW, RNodeTemplate #,FineScanRANOD, FineScanRANODEoverW


class PerformanceEvaluation(
    CoarseScanRANODEoverW,
):
    
    num_fine_scan = luigi.IntParameter(default=10)
    device = luigi.Parameter(default="cuda")
    
    def requires(self):

        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for fine_scan_index in range(self.num_fine_scan):
            model_list[f"model_{fine_scan_index}"] = [RNodeTemplate.req(self, w_value=w_range[fine_scan_index], train_random_seed=(i)) for i in range(self.num_sig_templates)]

        return {
            "models": model_list,
            "scan_result": CoarseScanRANODEoverW.req(self),
            "test_data": Preprocessing.req(self),
            "bkgprob": PredictBkgProb.req(self),
        }

    def output(self):
        return {
            "performance_plot": self.local_target("performance_plot.pdf"),
            "sic_values": self.local_target("sic_values.json"),
        }
    
    @law.decorator.safe_output
    def run(self):

        # load the best models
        with open(self.input()["scan_result"]["peak_info"].path, 'r') as f:
            scan_result = json.load(f)

        best_model_index = scan_result["mu_best_index"]

        model_best_list = []
        model_loss_list = []

        for rand_seed_index in range(self.num_sig_templates):
            model_best_seed_i = self.input()["models"][f"model_{best_model_index}"][rand_seed_index]["sig_models"]
            metadata_best_seed_i = self.input()["models"][f"model_{best_model_index}"][rand_seed_index]["metadata"].load()

            for model in model_best_seed_i:
                model_best_list.append(model.path)
            model_loss_list.extend(metadata_best_seed_i["min_val_loss_list"])

        # select 20 best models
        model_loss_list = np.array(model_loss_list)
        model_loss_list = np.sort(model_loss_list)[:20]
        model_best_list = model_best_list[:20]

        # load test data
        data_test_SR_model_S = np.load(self.input()['test_data']['data_test_SR_model_S'].path)

        # load bkg prob
        data_test_SR_prob_B = np.load(self.input()['bkgprob']['log_B_test'].path)
        data_test_SR_prob_B = np.exp(data_test_SR_prob_B.flatten())

        from src.models.train_model_S import pred_model_S

        prob_S_list = []

        for model_dir in model_best_list:
            prob_S = pred_model_S(model_dir, data_test_SR_model_S, batch_size=2048, device=self.device)
            prob_S_list.append(prob_S)

        prob_S_list = np.array(prob_S_list)
        prob_S_list = np.mean(prob_S_list, axis=0)

        prob_anomaly = prob_S_list / (1e-10 + data_test_SR_prob_B) + 1e-10
        prob_anomaly = np.log(prob_anomaly)
        # adjust to 0-1
        prob_anomaly = (prob_anomaly - prob_anomaly.min()) / (prob_anomaly.max() - prob_anomaly.min())

        truth_label = data_test_SR_model_S[:, -1].flatten()

        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(truth_label, prob_anomaly)
        sic = tpr / np.sqrt(fpr)

        # fine sic curve at fpr = 0.001
        arg_fpr = np.argmin(np.abs(fpr - 0.001))
        sic_value = sic[arg_fpr]

        self.output()["performance_plot"].parent.touch()

        with PdfPages(self.output()["performance_plot"].path) as pdf:
            bins=np.linspace(0, 1, 100)
            f = plt.figure()
            plt.hist(prob_anomaly[truth_label == 0], bins=bins, label='bkg', density=True, histtype='step', lw=3)
            plt.hist(prob_anomaly[truth_label == 1], bins=bins, label='sig', density=True, histtype='step', lw=3)
            plt.xlabel('anomaly score')
            plt.ylabel('num events')
            plt.title('Anomaly score distribution')
            plt.legend()
            plt.yscale('log')
            pdf.savefig(f)
            plt.close(f)

            f = plt.figure()
            plt.plot(tpr, sic, label='SIC')
            plt.xlabel('TPR')
            plt.ylabel('SIC')
            plt.title('SIC vs TPR')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

        with open(self.output()["sic_values"].path, 'w') as f:
            json.dump({"sic_value_fpr001": sic_value}, f, cls=NumpyEncoder)


# class ScanOverTruthMu(
#     FineScanRANODEoverW,
# ):
    
#     def requires(self):
#         truth_mu_scan_list = [0.0005, 0.001, 0.005, 0.01]
    
#         return {
#             "fine_scan_result": [FineScanRANODEoverW.req(self, s_ratio=truth_mu) for truth_mu in truth_mu_scan_list],
#             "sic_result": [PerformanceEvaluation.req(self, s_ratio=truth_mu) for truth_mu in truth_mu_scan_list],
#         }
    
#     def output(self):
#         return self.local_target("scan_plot.pdf")
    
#     @law.decorator.safe_output
#     def run(self):

#         truth_mu_scan_list = [0.0005, 0.001, 0.005, 0.01]

#         mu_pred_list = []
#         mu_lowerbound_list = []
#         mu_upperbound_list = []
#         sic_values = []

#         for index, truth_mu in enumerate(truth_mu_scan_list):

#             fine_scan_result = json.load(open(self.input()["fine_scan_result"][index]["scan_result"].path, 'r'))
#             sic_value = json.load(open(self.input()["sic_result"][index]["sic_values"].path, 'r'))["sic_value_fpr001"]

#             mu_pred_i = fine_scan_result["mu_pred"]
#             mu_lowerbound_i = fine_scan_result["mu_lowerbound"]
#             mu_upperbound_i = fine_scan_result["mu_upperbound"]

#             mu_pred_list.append(mu_pred_i)
#             mu_lowerbound_list.append(mu_lowerbound_i)
#             mu_upperbound_list.append(mu_upperbound_i)
#             sic_values.append(sic_value)

#         # plot
#         with PdfPages(self.output().path) as pdf:
#             f = plt.figure()
#             plt.plot(truth_mu_scan_list, mu_pred_list, label='pred $\mu$', color='red')
#             plt.fill_between(truth_mu_scan_list, mu_lowerbound_list, mu_upperbound_list, alpha=0.2, color='red')
#             plt.plot(truth_mu_scan_list, truth_mu_scan_list, label='true $\mu$', color='black')
#             plt.xlabel('true $\mu$')
#             plt.ylabel('pred $\mu$')
#             plt.xscale('log')
#             plt.yscale('log')
#             plt.title('Scan over truth $\mu$')
#             plt.legend()
#             pdf.savefig(f)
#             plt.close(f)

#             f = plt.figure()
#             plt.plot(truth_mu_scan_list, sic_values, label='SIC', color='red')
#             plt.xlabel('true $\mu$')
#             plt.ylabel('SIC')
#             plt.xscale('log')
#             plt.title('SIC vs true $\mu$')
#             plt.legend()
#             pdf.savefig(f)
#             plt.close(f)

        

