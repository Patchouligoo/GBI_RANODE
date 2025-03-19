import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    TranvalSplitRandomMixin,
    TemplateRandomMixin,
    TranvalSplitUncertaintyMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    TestSetMixin,
    WScanMixin,
    BkgModelMixin,
)
from src.tasks.preprocessing import PreprocessingTrainval, PreprocessingTest
from src.tasks.bkgtemplate import PredictBkgProbTrainVal, PredictBkgProbTest
from src.utils.utils import NumpyEncoder, str_encode_value


class RNodeTemplate(
    TranvalSplitRandomMixin,
    TemplateRandomMixin,
    BkgModelMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    device = luigi.Parameter(default="cuda:0")
    batchsize = luigi.IntParameter(default=2048)
    epoches = luigi.IntParameter(default=200)
    w_value = luigi.FloatParameter(default=0.05)
    early_stopping_patience = luigi.IntParameter(default=10)

    def store_parts(self):
        w_value = str(self.w_value)
        return super().store_parts() + (f"w_{w_value}",)

    def requires(self):
        return {
            "preprocessed_data": PreprocessingTrainval.req(
                self,
                trainval_split_seed=self.trainval_split_seed,
                s_ratio_index=self.s_ratio_index,
            ),
            "bkgprob": PredictBkgProbTrainVal.req(
                self,
                trainval_split_seed=self.trainval_split_seed,
                s_ratio_index=self.s_ratio_index,
            ),
        }

    def output(self):
        return {
            "sig_model": self.local_target(f"model_S.pt"),
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
            "metadata": self.local_target("metadata.json"),
        }

    @law.decorator.safe_output
    def run(self):

        input_dict = {
            "preprocessing": {
                "data_train_SR_model_S": self.input()["preprocessed_data"][
                    "SR_data_train_model_S"
                ],
                "data_val_SR_model_S": self.input()["preprocessed_data"][
                    "SR_data_val_model_S"
                ],
                "data_train_SR_model_B": self.input()["preprocessed_data"][
                    "SR_data_train_model_B"
                ],
                "data_val_SR_model_B": self.input()["preprocessed_data"][
                    "SR_data_val_model_B"
                ],
                "SR_mass_hist": self.input()["preprocessed_data"]["SR_mass_hist"],
            },
            "bkgprob": {
                "log_B_train": self.input()["bkgprob"]["log_B_train"],
                "log_B_val": self.input()["bkgprob"]["log_B_val"],
            },
        }

        print(
            f"train model S with train random seed {self.train_random_seed}, sample random seed {self.trainval_split_seed}, s_ratio {self.s_ratio}"
        )
        from src.models.train_model_S import train_model_S

        train_model_S(
            input_dict,
            self.output(),
            self.s_ratio,
            self.w_value,
            self.batchsize,
            self.epoches,
            self.early_stopping_patience,
            self.train_random_seed,
            self.device,
        )


class CoarseScanRANODEFixedSplitSeed(
    SigTemplateTrainingUncertaintyMixin,
    TranvalSplitRandomMixin,
    WScanMixin,
    BkgModelMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):

        model_list = {}
        w_range = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [
                RNodeTemplate.req(
                    self,
                    w_value=w_range[index],
                    trainval_split_seed=self.trainval_split_seed,
                    train_random_seed=i,
                )
                for i in range(self.train_num_sig_templates)
            ]

        return model_list

    def output(self):
        return {
            "coarse_scan_plot": self.local_target("fitting_result.pdf"),
            "scan_result": self.local_target("scan_result.json"),
            "model_list": self.local_target("model_list.json"),
        }

    @law.decorator.safe_output
    def run(self):

        w_range = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )
        w_range_log = np.log10(w_range)

        val_loss_scan = []
        model_path_list_scan = {}

        for index_w in range(self.scan_number):

            val_loss_list = []
            model_path_list = []

            for i in range(self.train_num_sig_templates):

                # save min val loss
                metadata_w_i = self.input()[f"model_{index_w}"][i]["metadata"].load()
                min_val_loss_list = metadata_w_i["min_val_loss_list"]
                val_events_num = metadata_w_i["num_val_events"]
                val_loss_list.extend(min_val_loss_list)

                # save model paths
                model_path_list_i = [
                    self.input()[f"model_{index_w}"][i]["sig_model"].path
                ]
                model_path_list.extend(model_path_list_i)

            val_loss_scan.append(val_loss_list)
            model_path_list_scan[f"scan_index_{index_w}"] = model_path_list

        val_loss_scan = np.array(val_loss_scan)
        val_loss_scan = -1 * val_loss_scan

        val_loss_scan_mean = np.mean(val_loss_scan, axis=1)
        val_loss_scan_std = np.std(val_loss_scan, axis=1)

        from src.fitting.fitting import fit_likelihood

        self.output()["coarse_scan_plot"].parent.touch()
        output_metadata = fit_likelihood(
            w_range_log,
            val_loss_scan_mean,
            val_loss_scan_std,
            np.log10(self.s_ratio),
            val_events_num,
            self.output()["coarse_scan_plot"].path,
        )

        with open(self.output()["scan_result"].path, "w") as f:
            json.dump(output_metadata, f, cls=NumpyEncoder)

        with open(self.output()["model_list"].path, "w") as f:
            json.dump(model_path_list_scan, f, cls=NumpyEncoder)


class CoarseScanRANODEoverW(
    SigTemplateTrainingUncertaintyMixin,
    TranvalSplitUncertaintyMixin,
    BkgModelMixin,
    TestSetMixin,
    WScanMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):

        trainval_seed_results = {}

        for index in range(1, self.split_num_sig_templates + 1):  # start from 1
            trainval_seed_results[f"trainval_seed_{index}"] = (
                CoarseScanRANODEFixedSplitSeed.req(self, trainval_split_seed=index)
            )

        return {
            "model_S_scan_result": trainval_seed_results,
            "test_data": PreprocessingTest.req(self),
            "bkgprob_test": PredictBkgProbTest.req(self),
        }

    def output(self):
        return {
            "prob_S_scan": self.local_target("prob_S_scan.npy"),
            "prob_B_scan": self.local_target("prob_B_scan.npy"),
        }

    @law.decorator.safe_output
    def run(self):

        # load model list
        model_scan_dict = {
            f"scan_index_{index}": [] for index in range(self.scan_number)
        }
        for index in range(self.train_num_sig_templates):
            with open(
                self.input()["model_S_scan_result"][f"trainval_seed_{index}"][
                    "model_list"
                ].path,
                "r",
            ) as f:
                model_list = json.load(f)
                for key, value in model_list.items():
                    model_scan_dict[key].extend(value)

        # load test data
        test_data = self.input()["test_data"]
        truth_label = np.load(test_data["SR_data_test_model_S"].path)[:, -1]

        # load bkg prob
        bkg_prob = self.input()["bkgprob_test"]["log_B_test"]
        event_num = np.load(bkg_prob.path).shape[0]

        from src.models.ranode_pred import ranode_pred

        w_scan_list = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )
        prob_S_list = []
        prob_B_list = []

        for w_index in range(self.scan_number):
            w_value = w_scan_list[w_index]

            print(f"evaluating scan index {w_index}, w value {w_value}")

            model_list = model_scan_dict[f"scan_index_{w_index}"]
            prob_S, prob_B = ranode_pred(model_list, w_value, test_data, bkg_prob)

            prob_S_list.append(prob_S)
            prob_B_list.append(prob_B)

        prob_S_list = np.array(prob_S_list)
        prob_B_list = np.array(prob_B_list)

        self.output()["prob_S_scan"].parent.touch()
        np.save(self.output()["prob_S_scan"].path, prob_S_list)
        np.save(self.output()["prob_B_scan"].path, prob_B_list)


# class FineScanRANOD(
#     TemplateRandomMixin,
#     SignalStrengthMixin,
#     ProcessMixin,
#     BaseTask,
# ):

#     fine_scan_index = luigi.IntParameter(default=0)
#     num_fine_scan = luigi.IntParameter(default=10)
#     batchsize = luigi.IntParameter(default=2048)
#     epoches = luigi.IntParameter(default=100)
#     num_model_to_save = luigi.IntParameter(default=10)
#     device = luigi.Parameter(default="cuda")

#     def store_parts(self):
#         return super().store_parts() + (f"fine_scan_{self.fine_scan_index}",)

#     def requires(self):
#         return {
#             'coarse_scan': CoarseScanRANODEoverW.req(self),
#             'preprocessing': Preprocessing.req(self),
#             'bkgprob': PredictBkgProb.req(self),
#         }

#     def output(self):
#         return {
#             "sig_models": [self.local_target(f"model_S_{i}.pt") for i in range(self.num_model_to_save)],
#             "trainloss_list": self.local_target("trainloss_list.npy"),
#             "valloss_list": self.local_target("valloss_list.npy"),
#             "metadata": self.local_target("metadata.json"),
#         }

#     @law.decorator.safe_output
#     def run(self):

#         with open(self.input()["coarse_scan"]["peak_info"].path, 'r') as f:
#             peak_info = json.load(f)

#         mu_lower = peak_info["mu_fine_scan_range_left"]
#         mu_upper = peak_info["mu_fine_scan_range_right"]

#         mu_scan_range = np.linspace(mu_lower, mu_upper, self.num_fine_scan+2)[1:-1]

#         curr_mu = mu_scan_range[self.fine_scan_index]

#         print("scan over ", mu_scan_range)
#         print("current value ", curr_mu)

#         from src.models.train_model_S import train_model_S
#         train_model_S(self.input(), self.output(), self.s_ratio, curr_mu, self.batchsize, self.epoches, self.num_model_to_save, self.train_random_seed, self.device)


# class FineScanRANODEoverW(
#     SigTemplateUncertaintyMixin,
#     SignalStrengthMixin,
#     ProcessMixin,
#     BaseTask,
# ):

#     num_fine_scan = luigi.IntParameter(default=10)

#     def requires(self):

#         model_list = {}
#         for fine_scan_index in range(self.num_fine_scan):
#             model_list[f"fine_scan_{fine_scan_index}"] = [FineScanRANOD.req(self, fine_scan_index=fine_scan_index, train_random_seed=(i+100)) for i in range(self.num_sig_templates)]

#         return {
#             "fine_scan_models": model_list,
#             "coarse_scan": CoarseScanRANODEoverW.req(self),
#         }

#     def output(self):
#         return {
#             "scan_result": self.local_target("scan_result.json"),
#             "fine_scan_plot": self.local_target("fitting_result.pdf"),
#         }

#     @law.decorator.safe_output
#     def run(self):

#         with open(self.input()["coarse_scan"]["peak_info"].path, 'r') as f:
#             peak_info = json.load(f)

#         mu_lower = peak_info["mu_fine_scan_range_left"]
#         mu_upper = peak_info["mu_fine_scan_range_right"]
#         mu_fine_scan_range = np.linspace(mu_lower, mu_upper, self.num_fine_scan+2)[1:-1]

#         mu_dp_coarse_scan = peak_info["mu_dp_coarse_scan"]
#         mu_dp_mean_coarse_scan = peak_info["mu_dp_mean_coarse_scan"]
#         mu_dp_std_coarse_scan = peak_info["mu_dp_std_coarse_scan"]

#         val_loss_scan = []

#         for index_w in range(self.num_fine_scan):

#             val_loss_list = []

#             for index_seed in range(self.num_sig_templates):
#                 metadata_w_i = self.input()["fine_scan_models"][f"fine_scan_{index_w}"][index_seed]["metadata"].load()
#                 min_val_loss_list = metadata_w_i["min_val_loss_list"]
#                 val_events_num = metadata_w_i["num_val_events"]
#                 val_loss_list.extend(min_val_loss_list)

#             val_loss_scan.append(val_loss_list)

#         # pick the top 20 models with smallest loss
#         val_loss_scan = np.array(val_loss_scan)
#         # val_loss_scan = np.sort(val_loss_scan, axis=-1)[:, :20]

#         # multiple by -1 since the loss is -log[mu*P(sig) + (1-mu)*P(bkg)] but we want likelihood
#         # which is log[mu*P(sig) + (1-mu)*P(bkg)]
#         val_loss_scan = -1 * val_loss_scan
#         val_loss_scan_mean = np.mean(val_loss_scan, axis=1)
#         val_loss_scan_std = np.std(val_loss_scan, axis=1)

#         # combine coarse scan and fine scan in order of mu
#         mu_scan_range = np.concatenate([mu_dp_coarse_scan, mu_fine_scan_range])
#         val_loss_scan_mean = np.concatenate([mu_dp_mean_coarse_scan, val_loss_scan_mean])
#         val_loss_scan_std = np.concatenate([mu_dp_std_coarse_scan, val_loss_scan_std])

#         # sort the mu values
#         sort_index = np.argsort(mu_scan_range)
#         mu_scan_range = mu_scan_range[sort_index]
#         val_loss_scan_mean = val_loss_scan_mean[sort_index]
#         val_loss_scan_std = val_loss_scan_std[sort_index]

#         from src.fitting.fitting import fit_likelihood
#         self.output()["fine_scan_plot"].parent.touch()
#         output_metadata = fit_likelihood(np.log10(mu_scan_range), val_loss_scan_mean, val_loss_scan_std, np.log10(self.s_ratio), val_events_num, self.output()["fine_scan_plot"].path)

#         mu_pred = output_metadata["mu_pred"]
#         best_model_index_fine_scan = np.argmin(np.abs(mu_fine_scan_range - mu_pred))
#         mu_lowerbound = output_metadata["mu_lowerbound"]
#         mu_upperbound = output_metadata["mu_upperbound"]

#         # scan result
#         scan_result = {
#             "mu_true": self.s_ratio,
#             "mu_pred": mu_pred,
#             "best_model_index_fine_scan": best_model_index_fine_scan,
#             "mu_lowerbound": mu_lowerbound,
#             "mu_upperbound": mu_upperbound,
#         }

#         with open(self.output()["scan_result"].path, 'w') as f:
#             json.dump(scan_result, f, cls=NumpyEncoder)


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
