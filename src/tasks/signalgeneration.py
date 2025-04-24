import os, sys
import importlib
import luigi
import law
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    WScanMixin,
    BkgModelMixin,
)
from src.tasks.preprocessing import PreprocessingFold, ProcessBkg, ProcessSignal
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.bkgsampling import PredictBkgProbGen, PreprocessingFoldwModelBGen
from src.tasks.rnodetemplate import (
    ScanRANODE,
    RNodeTemplate,
    ScanRANODEFixedSeed,
)


class ProcessAllSignals(
    BaseTask
):
    
    mx = luigi.IntParameter(default=100)
    my = luigi.IntParameter(default=500)

    def store_parts(self):
        return super().store_parts() + (
            f"mx_{self.mx}",
            f"my_{self.my}",
        )
    
    def output(self):
        return {
            "signals": self.local_target("reprocessed_signals.npy"),
        }
    
    @law.decorator.safe_output
    def run(self):
        data_dir = os.environ.get("DATA_DIR")

        data_path = f"{data_dir}/extra_raw_lhco_samples/events_anomalydetection_Z_XY_qq_parametric.h5"

        from src.data_prep.signal_processing import process_raw_signals
        self.output()["signals"].parent.touch()
        output_path = self.output()["signals"].path
        process_raw_signals(data_path, output_path, self.mx, self.my)


class SignalGeneration(
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    WScanMixin,
    SignalStrengthMixin,
    BaseTask,
):

    w_test_index = luigi.IntParameter(default=0)
    device = luigi.Parameter(default="cuda:0")
    num_generated_sigs = luigi.IntParameter(default=1000000)

    mx = luigi.IntParameter(default=100)
    my = luigi.IntParameter(default=500)

    num_ensembles = luigi.IntParameter(default=5)

    def store_parts(self):
      w_test_value = self.w_range[self.w_test_index]
      return super().store_parts() + (
            f"mx_{self.mx}",
            f"my_{self.my}",
            f"num_ensembles_{self.num_ensembles}",
            f"w_test_index_{self.w_test_index}_value_{str_encode_value(w_test_value)}",
        )

    def requires(self):
        model_results = {}

        for ensemble_index in range(self.num_ensembles):
            model_results_ensemble_i = {}
            for index in range(self.train_num_sig_templates):
                model_results_ensemble_i[f"model_seed_{index}"] = ScanRANODEFixedSeed.req(
                    self, train_random_seed=index,
                )

            model_results[f"ensemble_{ensemble_index}"] = model_results_ensemble_i

        return {
            "model_S_scan_result": model_results,
            "preprocessing_params": ProcessBkg.req(self),
        }

    def output(self):
        return self.local_target("generated_signals.npy")

    @law.decorator.safe_output
    def run(self):

        # load the preprocessing parameters
        pre_parameters = json.load(
            open(self.input()["preprocessing_params"]["pre_parameters"].path, "r")
        )

        # ------------------- generate the signal events -------------------
        # load models
        model_results = self.input()["model_S_scan_result"]
        model_list = []

        for ensemble_index in range(self.num_ensembles):
            model_results_ensemble_i = model_results[f"ensemble_{ensemble_index}"]
            for model_rand_index in range(self.train_num_sig_templates):
                model_results_ensemble_i_seed_j_path = model_results_ensemble_i[f"model_seed_{model_rand_index}"][
                    "model_list"
                ].path
                model_results_ensemble_i_seed_j = json.load(open(model_results_ensemble_i_seed_j_path, "r"))

                model_list.append(model_results_ensemble_i_seed_j[f"scan_index_{self.w_test_index}"][0])

        # generated events using each model
        num_models = len(model_list)
        print(f"Sampling {self.num_generated_sigs} events from {num_models} models")
        from src.models.model_S import flows_model_RQS

        generated_events = []
        for model_path in tqdm(model_list):
            model_S = flows_model_RQS(
                device=self.device, num_features=5, context_features=None
            )

            model_S.load_state_dict(torch.load(model_path, weights_only=True))
            model_S.eval()

            with torch.no_grad():
                sampled_signals = model_S.sample(
                    num_samples=int(
                        self.num_generated_sigs / num_models,
                    )
                )

            generated_events.append(sampled_signals.cpu().numpy())
        generated_events = np.concatenate(generated_events, axis=0)

        # need to use pre_params to undo the normalization
        from src.data_prep.utils import inverse_transform

        # need to manually add label to the generated events
        generated_events_label = np.ones(len(generated_events))
        generated_events = np.concatenate(
            [generated_events, generated_events_label[:, None]], axis=1
        )
        generated_events = inverse_transform(generated_events, pre_parameters)
        # for mass, need to add 3.5 to generated signals due to preprocessing procedure
        generated_events[:, 0] += 3.5

        # ------------------- save the generated events -------------------
        self.output().parent.touch()
        np.save(self.output().path, generated_events)


class SignalGenerationPlot(SignalGeneration):

    def requires(self):
        return {
            "generated_signals": SignalGeneration.req(self),
            "bkg_events": ProcessBkg.req(self),
            "real_sig": ProcessAllSignals.req(self),
        }

    def output(self):
        return {
            "generated_features": self.local_target("comparison_plots.pdf"),
            "generated_m1m2": self.local_target("comparison_m1m2.pdf"),
        }

    @law.decorator.safe_output
    def run(self):
        feature_list = ["mjj", "mjmin", "mjmax - mjmin", "tau21min", "tau21max"]

        # load bkg events
        bkg_file = self.input()["bkg_events"]["SR_bkg"].path
        bkg_events = np.load(bkg_file)[:, :-1]  # remove the label column
        bkg_df = pd.DataFrame(bkg_events, columns=feature_list)

        # load the real signal events
        real_sig_file = self.input()["real_sig"]["signals"].path
        real_sig_events = np.load(real_sig_file)[:, :-1]  # remove the label column
        real_sig_df = pd.DataFrame(real_sig_events, columns=feature_list)

        # load the generated events
        generated_file = self.input()["generated_signals"].path
        generated_events = np.load(generated_file)[:, :-1]  # remove the label column
        generated_df = pd.DataFrame(generated_events, columns=feature_list)

        # make plots
        dfs = {
            "real_signals": real_sig_df,
            "generated_signals": generated_df,
            "background": bkg_df,
        }       
        
        plot_options = {
            "real_signals": {
                "styles": {
                    "color": "black",
                    "ls": "-",
                    "lw": 2,
                }
            },
            "generated_signals": {
                "styles": {
                    "color": "red",
                    "ls": "-",
                    "lw": 3,
                }
            },
            "background": {
                "styles": {
                    "color": "blue",
                    "ls": "--",
                    "lw": 1,
                }
            },
        }

        metadata = {
            "mx": self.mx,
            "my": self.my,
            "mu_true": self.s_ratio,
            "numB": len(bkg_df),
            "mu_test": self.w_range[self.w_test_index],
            "use_full_stats": self.use_full_stats,
            "use_perfect_modelB": self.use_perfect_bkg_model,
            "use_modelB_genData": self.use_bkg_model_gen_data,
            "columns": feature_list,
        }
        self.output()["generated_features"].parent.touch()
        from src.plotting.plotting import plot_event_feature_distribution
        plot_event_feature_distribution(dfs, metadata, plot_options, self.output()["generated_features"].path)


        # make m1m2 plots
        bkg_mjmin = bkg_df["mjmin"].values
        bkg_mjmax = bkg_df["mjmax - mjmin"].values + bkg_mjmin
        bkg_m1m2 = np.concatenate([bkg_mjmin, bkg_mjmax], axis=0) * 1000
        bkg_m1m2_df = pd.DataFrame(bkg_m1m2, columns=["m1 m2 (GeV)"])

        real_sig_mjmin = real_sig_df["mjmin"].values
        real_sig_mjmax = real_sig_df["mjmax - mjmin"].values + real_sig_mjmin
        real_sig_m1m2 = np.concatenate([real_sig_mjmin, real_sig_mjmax], axis=0) * 1000
        real_sig_m1m2_df = pd.DataFrame(real_sig_m1m2, columns=["m1 m2 (GeV)"])

        generated_mjmin = generated_df["mjmin"].values
        generated_mjmax = generated_df["mjmax - mjmin"].values + generated_mjmin
        generated_m1m2 = np.concatenate([generated_mjmin, generated_mjmax], axis=0) * 1000
        generated_m1m2_df = pd.DataFrame(generated_m1m2, columns=["m1 m2 (GeV)"])

        m1m2_dfs = {
            "real_signals": real_sig_m1m2_df,
            "generated_signals": generated_m1m2_df,
            "background": bkg_m1m2_df,
        }

        metadata = {
            "mx": self.mx,
            "my": self.my,
            "mu_true": self.s_ratio,
            "numB": len(bkg_df),
            "mu_test": self.w_range[self.w_test_index],
            "use_full_stats": self.use_full_stats,
            "use_perfect_modelB": self.use_perfect_bkg_model,
            "use_modelB_genData": self.use_bkg_model_gen_data,
            "columns": ["m1 m2 (GeV)"],
        }

        plot_event_feature_distribution(
            m1m2_dfs,
            metadata,
            plot_options,
            self.output()["generated_m1m2"].path,
        )

        # law run SignalGenerationPlot --version dev_smallS_10k_all --use-full-stats True --use-perfect-bkg-model True --use-bkg-model-gen-data False --s-ratio-index 7 --w-test-index 11
