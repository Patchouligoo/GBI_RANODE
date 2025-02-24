import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json

from src.utils.law import BaseTask, SignalStrengthMixin, TemplateRandomMixin, SigTemplateUncertaintyMixin, ProcessMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.rnodetemplate import FineScanRANOD, FineScanRANODEoverW


class PerformanceEvaluation(
    SigTemplateUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    num_fine_scan = luigi.IntParameter(default=10)
    device = luigi.Parameter(default="cuda")
    
    def requires(self):

        model_list = {}
        for fine_scan_index in range(self.num_fine_scan):
            model_list[f"fine_scan_{fine_scan_index}"] = [FineScanRANOD.req(self, fine_scan_index=fine_scan_index, train_random_seed=(i+42)) for i in range(self.num_sig_templates)]

        return {
            "fine_scan_models": model_list,
            "fine_scan": FineScanRANODEoverW.req(self),
            "test_data": Preprocessing.req(self),
            "bkgprob": PredictBkgProb.req(self),
        }

    def output(self):
        return {
            "performance_plot": self.local_target("performance_plot.pdf"),
        }
    
    @law.decorator.safe_output
    def run(self):

        # load the best models
        with open(self.input()["fine_scan"]["scan_result"].path, 'r') as f:
            scan_result = json.load(f)

        best_model_index = scan_result["best_model_index"]

        model_best_list = []
        model_loss_list = []

        for rand_seed_index in range(self.num_sig_templates):
            model_best_seed_i = self.input()["fine_scan_models"][f"fine_scan_{best_model_index}"][rand_seed_index]["sig_models"]
            metadata_best_seed_i = self.input()["fine_scan_models"][f"fine_scan_{best_model_index}"][rand_seed_index]["metadata"].load()

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

        # plot
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

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