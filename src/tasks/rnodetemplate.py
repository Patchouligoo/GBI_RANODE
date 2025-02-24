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

class RNodeTemplate(
    TemplateRandomMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    device = luigi.Parameter(default="cuda:0")
    batchsize = luigi.IntParameter(default=2048)
    epoches = luigi.IntParameter(default=100)
    w_value = luigi.FloatParameter(default=0.05)
    num_model_to_save = luigi.IntParameter(default=10)

    def store_parts(self):
        w_value = str(self.w_value)
        return super().store_parts() + (f"w_{w_value}",)

    def requires(self):
        return {
            'preprocessing': Preprocessing.req(self),
            'bkgprob': PredictBkgProb.req(self),
        }

    def output(self):
        return {
            "sig_models": [self.local_target(f"model_S_{i}.pt") for i in range(self.num_model_to_save)],
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
            "metadata": self.local_target("metadata.json"),
        }
    
    @law.decorator.safe_output 
    def run(self):
        from src.models.train_model_S import train_model_S
        train_model_S(self.input(), self.output(), self.s_ratio, self.w_value, self.batchsize, self.epoches, self.num_model_to_save, self.train_random_seed, self.device)        


class CoarseScanRANODEoverW(
    SigTemplateUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    w_min = luigi.FloatParameter(default=0.0001)
    w_max = luigi.FloatParameter(default=0.05)
    scan_number = luigi.IntParameter(default=10)

    def requires(self):

        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_sig_templates)]

        return model_list
    
    def output(self):
        return {
            "coarse_scan_plot": self.local_target("fitting_result.pdf"),
            "peak_info": self.local_target("peak_info.json"),
        }
    
    @law.decorator.safe_output
    def run(self):

        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)
        w_range_log = np.log10(w_range)

        val_loss_scan = []

        for index_w in range(self.scan_number):

            val_loss_list = []

            for index_seed in range(self.num_sig_templates):
                metadata_w_i = self.input()[f"model_{index_w}"][index_seed]["metadata"].load()
                min_val_loss_list = metadata_w_i["min_val_loss_list"]
                val_events_num = metadata_w_i["num_val_events"]
                val_loss_list.extend(min_val_loss_list)

            val_loss_scan.append(val_loss_list)

        # pick the top 20 models with smallest loss
        val_loss_scan = np.array(val_loss_scan)
        val_loss_scan = np.sort(val_loss_scan, axis=-1)[:, :20]

        # multiple by -1 since the loss is -log[mu*P(sig) + (1-mu)*P(bkg)] but we want likelihood
        # which is log[mu*P(sig) + (1-mu)*P(bkg)]
        val_loss_scan = -1 * val_loss_scan
        val_loss_scan_mean = np.mean(val_loss_scan, axis=1)
        val_loss_scan_std = np.std(val_loss_scan, axis=1)

        from src.fitting.fitting import fit_likelihood
        self.output()["coarse_scan_plot"].parent.touch()
        output_metadata = fit_likelihood(w_range_log, val_loss_scan_mean, val_loss_scan_std, np.log10(self.s_ratio), val_events_num, self.output()["coarse_scan_plot"].path)

        mu_pred = output_metadata["mu_pred"]

        # find the w test value closest to the peak likelihood
        w_best_index = np.argmin(np.abs(w_range - mu_pred))
        w_best = w_range[w_best_index]
        # find the index - 1, index + 1 w values, return boundary w values if index is at the boundary
        w_fine_scane_range_left = w_range[max(0, w_best_index-1)]
        w_fine_scane_range_right = w_range[min(self.scan_number-1, w_best_index+1)]

        peak_info = {
            "mu_true": self.s_ratio,
            "mu_pred": mu_pred,
            "mu_best": w_best,
            "mu_fine_scan_range_left": w_fine_scane_range_left,
            "mu_fine_scan_range_right": w_fine_scane_range_right,
        }

        with open(self.output()["peak_info"].path, 'w') as f:
            json.dump(peak_info, f, cls=NumpyEncoder)


class FineScanRANOD(
    TemplateRandomMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    fine_scan_index = luigi.IntParameter(default=0)
    num_fine_scan = luigi.IntParameter(default=10)
    batchsize = luigi.IntParameter(default=2048)
    epoches = luigi.IntParameter(default=100)
    num_model_to_save = luigi.IntParameter(default=10)
    device = luigi.Parameter(default="cuda")

    def store_parts(self):
        return super().store_parts() + (f"fine_scan_{self.fine_scan_index}",)

    def requires(self):
        return {
            'coarse_scan': CoarseScanRANODEoverW.req(self),
            'preprocessing': Preprocessing.req(self),
            'bkgprob': PredictBkgProb.req(self),
        }
    
    def output(self):
        return {
            "sig_models": [self.local_target(f"model_S_{i}.pt") for i in range(self.num_model_to_save)],
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
            "metadata": self.local_target("metadata.json"),
        } 
    
    @law.decorator.safe_output
    def run(self):

        with open(self.input()["coarse_scan"]["peak_info"].path, 'r') as f:
            peak_info = json.load(f)

        mu_lower = peak_info["mu_fine_scan_range_left"]
        mu_upper = peak_info["mu_fine_scan_range_right"]

        mu_scan_range = np.linspace(mu_lower, mu_upper, self.num_fine_scan+2)[1:-1]

        curr_mu = mu_scan_range[self.fine_scan_index]

        from src.models.train_model_S import train_model_S
        train_model_S(self.input(), self.output(), self.s_ratio, curr_mu, self.batchsize, self.epoches, self.num_model_to_save, self.train_random_seed, self.device)        


class FineScanRANODEoverW(
    SigTemplateUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    num_fine_scan = luigi.IntParameter(default=10)
    
    def requires(self):

        model_list = {}
        for fine_scan_index in range(self.num_fine_scan):
            model_list[f"fine_scan_{fine_scan_index}"] = [FineScanRANOD.req(self, fine_scan_index=fine_scan_index, train_random_seed=(i+100)) for i in range(self.num_sig_templates)]

        return {
            "fine_scan_models": model_list,
            "coarse_scan": CoarseScanRANODEoverW.req(self),
        }
    
    def output(self):
        return {
            "scan_result": self.local_target("scan_result.json"),
            "fine_scan_plot": self.local_target("fitting_result.pdf"),
        }
    
    @law.decorator.safe_output
    def run(self):
        
        with open(self.input()["coarse_scan"]["peak_info"].path, 'r') as f:
            peak_info = json.load(f)

        mu_lower = peak_info["mu_fine_scan_range_left"]
        mu_upper = peak_info["mu_fine_scan_range_right"]
        mu_scan_range = np.linspace(mu_lower, mu_upper, self.num_fine_scan+2)[1:-1]

        val_loss_scan = []

        for index_w in range(self.num_fine_scan):

            val_loss_list = []

            for index_seed in range(self.num_sig_templates):
                metadata_w_i = self.input()["fine_scan_models"][f"fine_scan_{index_w}"][index_seed]["metadata"].load()
                min_val_loss_list = metadata_w_i["min_val_loss_list"]
                val_events_num = metadata_w_i["num_val_events"]
                val_loss_list.extend(min_val_loss_list)

            val_loss_scan.append(val_loss_list) 

        # pick the top 20 models with smallest loss
        val_loss_scan = np.array(val_loss_scan)
        val_loss_scan = np.sort(val_loss_scan, axis=-1)[:, :20]

        # multiple by -1 since the loss is -log[mu*P(sig) + (1-mu)*P(bkg)] but we want likelihood
        # which is log[mu*P(sig) + (1-mu)*P(bkg)]
        val_loss_scan = -1 * val_loss_scan
        val_loss_scan_mean = np.mean(val_loss_scan, axis=1)
        val_loss_scan_std = np.std(val_loss_scan, axis=1)


        from src.fitting.fitting import fit_likelihood
        self.output()["fine_scan_plot"].parent.touch()
        output_metadata = fit_likelihood(mu_scan_range, val_loss_scan_mean, val_loss_scan_std, self.s_ratio, val_events_num, self.output()["fine_scan_plot"].path, logbased=False)

        mu_pred = output_metadata["mu_pred"]
        best_model_index = output_metadata["best_model_index"]
        mu_lowerbound = output_metadata["mu_lowerbound"]
        mu_upperbound = output_metadata["mu_upperbound"]

        # scan result
        scan_result = {
            "mu_true": self.s_ratio,
            "mu_pred": mu_pred,
            "best_model_index": best_model_index,
            "mu_lowerbound": mu_lowerbound,
            "mu_upperbound": mu_upperbound,
        }

        with open(self.output()["scan_result"].path, 'w') as f:
            json.dump(scan_result, f, cls=NumpyEncoder)

        
# class GenerateSignals(
#    ScanRANODEoverW, 
# ):
#     device = luigi.Parameter(default="cuda")
#     n_signal_samples = luigi.IntParameter(default=10000)
    
#     def requires(self):
#         model_list = {}
#         w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

#         for index in range(self.scan_number):
#             model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

#         w_scan_results = ScanRANODEoverW.req(self)

#         return {
#             "models": model_list,
#             "w_scan_results": w_scan_results,
#         } 

#     def output(self):
#         return {
#             "signal_list": self.local_target("signal_list.npy"),
#         }

#     @law.decorator.safe_output
#     def run(self):

#         # load previous scan result
#         w_scan_results = self.input()["w_scan_results"]["metadata"].load()
#         w_best = w_scan_results["w_best"]
#         w_best_index = w_scan_results["w_best_index"]

#         # load the model at best w
#         model_best_list = []
#         for rand_seed_index in range(len(self.input()["models"][f"model_{w_best_index}"])):
#             model_best_seed_i = self.input()["models"][f"model_{w_best_index}"][rand_seed_index]["sig_models"]
#             for model in model_best_seed_i:
#                 model_best_list.append(model.path)

#         # define model
#         ranode_path = os.environ.get("RANODE")
#         sys.path.append(ranode_path)
#         from nflow_utils import flows_model_RQS

#         # generate signals from each models
#         signal_list = []
#         for model_dir in model_best_list:
#             model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
#             model_S.load_state_dict(torch.load(model_dir))
#             model_S.eval()

#             signal_samples = model_S.sample(self.n_signal_samples)
#             signal_list.append(signal_samples.cpu().detach().numpy())

#             # clean cuda memory
#             del model_S
#             torch.cuda.empty_cache()

#         # sample weight is 1 / num_models
#         signal_list = np.array(signal_list)
#         sample_weight = 1 / signal_list.shape[0]
#         sample_weight = np.ones((len(signal_list), self.n_signal_samples, 1)) * sample_weight

#         signal_list = np.concatenate([signal_list, sample_weight], axis=-1)

#         signal_list = signal_list.reshape(-1, 6)

#         self.output()["signal_list"].parent.touch()
#         np.save(self.output()["signal_list"].path, signal_list)


# class SignalGenerationPlot(
#     GenerateSignals,
# ):
#     nbins = luigi.IntParameter(default=41)

#     def requires(self):
#         return {
#             "generated_signal_list": GenerateSignals.req(self, n_signal_samples=self.n_sig),
#             "preprocessing": Preprocessing.req(self),
#         }

#     def output(self):
#         return self.local_target("signal_plot.pdf")

#     @law.decorator.safe_output
#     def run(self):

#         generated_signals = np.load(self.input()["generated_signal_list"]["signal_list"].path)
#         generated_signal_features = generated_signals[:, 1:-1]
#         generated_signal_weights = generated_signals[:, -1]
#         generated_signals_mass = generated_signal_features[:, 0]

#         # load data
#         data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
#         mask_signals_val = data_val_SR_S[:, -1] == 1
#         signal_val = data_val_SR_S[mask_signals_val]
#         signal_val_features = signal_val[:, 1:-1]
#         signal_val_mass = signal_val_features[:, 0]

#         mask_bkg_val = data_val_SR_S[:, -1] == 0
#         bkg_val = data_val_SR_S[mask_bkg_val]
#         bkg_val_features = bkg_val[:, 1:-1]
#         bkg_val_mass = bkg_val_features[:, 0]

#         # plot
#         import matplotlib.pyplot as plt
#         from matplotlib.backends.backend_pdf import PdfPages

#         self.output().parent.touch()
#         with PdfPages(self.output().path) as pdf:
            
#             for feature_index in range(signal_val_features.shape[1]):

#                 bins = np.linspace(bkg_val_features[:, feature_index].min(), bkg_val_features[:, feature_index].max(), self.nbins)
                                   
#                 f = plt.figure()
#                 plt.hist(signal_val_features[:, feature_index], bins=bins, alpha=0.5, label='val signal', density=True, histtype='step', lw=3)
#                 plt.hist(bkg_val_features[:, feature_index], bins=bins, alpha=0.5, label='val bkg', density=True, histtype='step', lw=3)
#                 plt.hist(generated_signal_features[:, feature_index], bins=bins, weights=generated_signal_weights, 
#                          alpha=0.5, label='generated signal', density=True, histtype='step', lw=3)
#                 plt.xlabel(f'feature {feature_index}')
#                 plt.ylabel('density')
#                 plt.title(f'feature {feature_index} distribution, {self.n_sig} signal in samples')
#                 plt.legend()
#                 plt.yscale('log')
#                 pdf.savefig(f)
#                 plt.close(f)



