import os, sys
import importlib
import luigi
import law
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


class SignalGeneration(
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    WScanMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    w_test_index = luigi.IntParameter(default=0)
    device = luigi.Parameter(default="cuda:0")
    num_generated_sigs = luigi.IntParameter(default=1000000)

    def store_parts(self):
        w_test_value = self.w_range[self.w_test_index]
        return super().store_parts() + (
            f"w_test_index_{self.w_test_index}_value_{str_encode_value(w_test_value)}",
        )

    def requires(self):
        model_results = {}

        for index in range(self.train_num_sig_templates):
            model_results[f"model_seed_{index}"] = ScanRANODEFixedSeed.req(
                self, train_random_seed=index
            )

        return {
            "model_S_scan_result": model_results,
            "bkg": ProcessBkg.req(self),
            "real_sig": ProcessSignal.req(self),
        }

    def output(self):
        return self.local_target("generated_signals_comparison.pdf")

    @law.decorator.safe_output
    def run(self):
        # first load bkg events and preprocessing parameters
        bkg_file = self.input()["bkg"]["SR_bkg"].path
        bkg_events = np.load(bkg_file)
        num_bkg_events = bkg_events.shape[0]
        # load the preprocessing parameters
        pre_parameters = json.load(
            open(self.input()["bkg"]["pre_parameters"].path, "r")
        )
        # load the real signal events
        real_sig_file = self.input()["real_sig"]["signals"].path
        real_sig_events = np.load(real_sig_file)
        num_sig_events = real_sig_events.shape[0]

        # ------------------- generate the signal events -------------------
        # load models
        model_results = self.input()["model_S_scan_result"]
        model_list = []

        for model_rand_index in range(self.train_num_sig_templates):
            model_results_i_path = model_results[f"model_seed_{model_rand_index}"][
                "model_list"
            ].path
            model_results_i = json.load(open(model_results_i_path, "r"))

            model_list.append(model_results_i[f"scan_index_{self.w_test_index}"][0])

        # generated events using each model
        num_models = len(model_list)
        from src.models.model_S import flows_model_RQS

        generated_events = []
        for model_path in model_list:
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

        # make plots
        feature_list = ["mjj", "mjmin", "mjmax - mjmin", "tau21min", "tau21max"]
        self.output().parent.touch()
        with PdfPages(self.output().path) as pdf:
            for i in range(len(generated_events[0]) - 1):
                f = plt.figure(figsize=(10, 6))
                bins = np.linspace(
                    np.min(bkg_events[:, i]), np.max(bkg_events[:, i]), 100
                )
                plt.hist(
                    bkg_events[:, i],
                    bins=bins,
                    density=True,
                    label="bkg",
                    histtype="step",
                    color="black",
                )
                plt.hist(
                    generated_events[:, i],
                    bins=bins,
                    density=True,
                    label="generated sig",
                    histtype="step",
                    lw=3,
                    color="red",
                )
                plt.hist(
                    real_sig_events[:, i],
                    bins=bins,
                    density=True,
                    label="real sig",
                    histtype="step",
                    color="blue",
                )
                plt.xlabel(f"{feature_list[i]}")
                plt.ylabel("density")
                plt.legend()
                plt.title(
                    f"True mu = {self.s_ratio} \nmu_test={self.w_range[self.w_test_index]}"
                )
                pdf.savefig(f)
                plt.close(f)
