import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
from src.utils.utils import NumpyEncoder

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    ProcessMixin,
    TranvalSplitRandomMixin,
    BkginSRDataMixin,
    TestSetMixin,
)


class ProcessSignalTrainVal(
    TranvalSplitRandomMixin, SignalStrengthMixin, ProcessMixin, BaseTask
):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=1)
    """

    def output(self):
        return {
            "train": self.local_target("reprocessed_signals_train.npy"),
            "val": self.local_target("reprocessed_signals_val.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        data_dir = os.environ.get("DATA_DIR")
        data_path = f"{data_dir}/hopefully_really_final_signal_features_W_qq.h5"

        from src.data_prep.signal_processing import process_signals

        self.output()["train"].parent.touch()
        process_signals(
            data_path,
            self.output()["train"].path,
            self.mx,
            self.my,
            self.s_ratio,
            self.trainval_split_seed,
            type="x_train",
        )
        process_signals(
            data_path,
            self.output()["val"].path,
            self.mx,
            self.my,
            self.s_ratio,
            self.trainval_split_seed,
            type="x_val",
        )


class ProcessSignalTest(TestSetMixin, SignalStrengthMixin, ProcessMixin, BaseTask):
    def output(self):
        return {
            "test": self.local_target("reprocessed_signals_test.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        data_dir = os.environ.get("DATA_DIR")
        data_path = f"{data_dir}/hopefully_really_final_signal_features_W_qq.h5"

        from src.data_prep.signal_processing import process_signals_test

        self.output()["test"].parent.touch()
        process_signals_test(
            data_path,
            self.output()["test"].path,
            self.mx,
            self.my,
            self.s_ratio,
            self.test_set_fold,
            use_true_mu=self.use_true_mu,
        )


class ProcessBkg(BkginSRDataMixin, BaseTask):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=0)

    Bkg events will first be splitted into SR and CR, then SR will be splitted into trainval set and test set
    The overall CR will be used to calculate the normalizing parameters, then it will be applied on CR events
    Then CR events will be splitted into train and val set
    """

    def output(self):
        return {
            "SR_trainval": self.local_target("reprocessed_bkgs_trainval.npy"),
            "SR_test": self.local_target("reprocessed_bkgs_test.npy"),
            "CR_train": self.local_target("reprocessed_bkgs_cr_train.npy"),
            "CR_val": self.local_target("reprocessed_bkgs_cr_val.npy"),
            "pre_parameters": self.local_target("pre_parameters.json"),
        }

    @law.decorator.safe_output
    def run(self):
        data_dir = os.environ.get("DATA_DIR")

        data_path_qcd = f"{data_dir}/events_anomalydetection_v2.features.h5"
        data_path_extra_qcd = (
            f"{data_dir}/events_anomalydetection_qcd_extra_inneronly_features.h5"
        )

        from src.data_prep.bkg_processing import process_bkgs

        output_qcd = process_bkgs(data_path_qcd)
        output_extra_qcd = process_bkgs(data_path_extra_qcd)

        output_combined = np.concatenate([output_qcd, output_extra_qcd], axis=0)

        # split into trainval and test set
        from src.data_prep.data_prep import background_split

        SR_bkg_trainval, SR_bkg_test, CR_bkg = background_split(
            output_combined,
            bkg_num_in_sr_data=self.bkg_num_in_sr_data,
            resample_seed=42,
        )

        # save SR data
        self.output()["SR_trainval"].parent.touch()
        np.save(self.output()["SR_trainval"].path, SR_bkg_trainval)
        np.save(self.output()["SR_test"].path, SR_bkg_test)

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # ----------------------- calculate normalizing parameters -----------------------
        pre_parameters = preprocess_params_fit(CR_bkg)
        # save pre_parameters
        self.output()["pre_parameters"].parent.touch()
        with open(self.output()["pre_parameters"].path, "w") as f:
            json.dump(pre_parameters, f, cls=NumpyEncoder)

        # ----------------------- process data in CR -----------------------
        CR_bkg = preprocess_params_transform(CR_bkg, pre_parameters)
        CR_bkg_train, CR_bkg_val = train_test_split(
            CR_bkg, test_size=0.25, random_state=42
        )

        # save training and validation data in CR
        np.save(self.output()["CR_train"].path, CR_bkg_train)
        np.save(self.output()["CR_val"].path, CR_bkg_val)


class PreprocessingTrainval(
    TranvalSplitRandomMixin, SignalStrengthMixin, ProcessMixin, BaseTask
):
    """
    This task will take signal train val set with a given signal strength and train val set split index
    It also takes SR bkg trainval set, using the same train val split index as seed to split the SR bkg
    into train and val set

    Then it will mix the signal and bkg into SR data, and normalize the data using the normalizing parameters
    calculated from CR data
    """

    def requires(self):

        if self.s_ratio != 0:
            return {
                "signal": ProcessSignalTrainVal.req(
                    self, trainval_split_seed=self.trainval_split_seed
                ),
                "bkg": ProcessBkg.req(self),
            }
        else:
            return {
                "bkg": ProcessBkg.req(self),
            }

    def output(self):
        return {
            "SR_data_train_model_S": self.local_target(
                "data_SR_data_train_model_S.npy"
            ),
            "SR_data_val_model_S": self.local_target("data_SR_data_val_model_S.npy"),
            "SR_data_train_model_B": self.local_target(
                "data_SR_data_train_model_B.npy"
            ),
            "SR_data_val_model_B": self.local_target("data_SR_data_val_model_B.npy"),
            "SR_mass_hist": self.local_target("SR_mass_hist.json"),
        }

    @law.decorator.safe_output
    def run(self):

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # load data
        if self.s_ratio != 0:
            SR_signal_train = np.load(self.input()["signal"]["train"].path)
            SR_signal_val = np.load(self.input()["signal"]["val"].path)
        SR_bkg_trainval = np.load(self.input()["bkg"]["SR_trainval"].path)

        pre_parameters = json.load(
            open(self.input()["bkg"]["pre_parameters"].path, "r")
        )
        for key in pre_parameters.keys():
            pre_parameters[key] = np.array(pre_parameters[key])

        # ----------------------- mass hist in SR -----------------------
        from config.configs import SR_MIN, SR_MAX

        mass = SR_bkg_trainval[SR_bkg_trainval[:, -1] == 0, 0]
        bins = np.linspace(SR_MIN, SR_MAX, 50)
        hist_back = np.histogram(mass, bins=bins, density=True)
        # save mass histogram and bins
        self.output()["SR_mass_hist"].parent.touch()
        with open(self.output()["SR_mass_hist"].path, "w") as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # ----------------------- make SR data -----------------------
        SR_bkg_train, SR_bkg_val = train_test_split(
            SR_bkg_trainval, test_size=1 / 3, random_state=self.trainval_split_seed
        )

        if self.s_ratio != 0:
            SR_data_train = np.concatenate([SR_signal_train, SR_bkg_train], axis=0)
            SR_data_val = np.concatenate([SR_signal_val, SR_bkg_val], axis=0)
        else:
            SR_data_train = SR_bkg_train
            SR_data_val = SR_bkg_val

        SR_data_train = shuffle(SR_data_train, random_state=self.trainval_split_seed)
        SR_data_val = shuffle(SR_data_val, random_state=self.trainval_split_seed)

        # SR_data_trainval
        _, mask = logit_transform(
            SR_data_train[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
        )
        SR_data_train = SR_data_train[mask]
        SR_data_train = preprocess_params_transform(SR_data_train, pre_parameters)
        # here x_train will be feed into both model_S and model_B later, to get prob of signal and background

        # val data
        _, mask = logit_transform(
            SR_data_val[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
        )
        SR_data_val = SR_data_val[mask]
        SR_data_val = preprocess_params_transform(SR_data_val, pre_parameters)

        # For signal model, we shift the mass by -3.5 following RANODE workflow
        # copy one set for signal model
        SR_data_train_model_S = SR_data_train.copy()
        SR_data_val_model_S = SR_data_val.copy()
        # shift mass by -3.5 for signals
        SR_data_train_model_S[:, 0] -= 3.5
        SR_data_val_model_S[:, 0] -= 3.5

        np.save(self.output()["SR_data_train_model_S"].path, SR_data_train_model_S)
        np.save(self.output()["SR_data_val_model_S"].path, SR_data_val_model_S)

        # copy another set for background model
        SR_data_train_model_B = SR_data_train.copy()
        SR_data_val_model_B = SR_data_val.copy()

        np.save(self.output()["SR_data_train_model_B"].path, SR_data_train_model_B)
        np.save(self.output()["SR_data_val_model_B"].path, SR_data_val_model_B)


class PreprocessingTest(TestSetMixin, SignalStrengthMixin, ProcessMixin, BaseTask):
    """
    This task will take signal test set with a given signal strength and test set fold index
    """

    def requires(self):
        if self.s_ratio != 0:
            return {
                "signal": ProcessSignalTest.req(
                    self, test_set_fold=self.test_set_fold, use_true_mu=self.use_true_mu
                ),
                "bkg": ProcessBkg.req(self),
            }
        else:
            return {
                "bkg": ProcessBkg.req(self),
            }

    def output(self):
        return {
            "SR_data_test_model_S": self.local_target("data_SR_data_test_model_S.npy"),
            "SR_data_test_model_B": self.local_target("data_SR_data_test_model_B.npy"),
            "SR_mass_hist": self.local_target("SR_mass_hist.json"),
        }

    @law.decorator.safe_output
    def run(self):
        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # SR mass hist will be the same as Trainval task
        SR_bkg_trainval = np.load(self.input()["bkg"]["SR_trainval"].path)
        # ----------------------- mass hist in SR -----------------------
        from config.configs import SR_MIN, SR_MAX

        mass = SR_bkg_trainval[SR_bkg_trainval[:, -1] == 0, 0]
        bins = np.linspace(SR_MIN, SR_MAX, 50)
        hist_back = np.histogram(mass, bins=bins, density=True)
        # save mass histogram and bins
        self.output()["SR_mass_hist"].parent.touch()
        with open(self.output()["SR_mass_hist"].path, "w") as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # preprocessing parameters
        pre_parameters = json.load(
            open(self.input()["bkg"]["pre_parameters"].path, "r")
        )
        for key in pre_parameters.keys():
            pre_parameters[key] = np.array(pre_parameters[key])

        # load data
        if self.s_ratio != 0:
            SR_signal_test = np.load(self.input()["signal"]["test"].path)
        SR_bkg_test = np.load(self.input()["bkg"]["SR_test"].path)

        if self.s_ratio != 0:
            SR_data_test = np.concatenate([SR_signal_test, SR_bkg_test], axis=0)
        else:
            SR_data_test = SR_bkg_test

        SR_data_test = shuffle(SR_data_test, random_state=42)

        _, mask = logit_transform(
            SR_data_test[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
        )
        SR_data_test = SR_data_test[mask]
        SR_data_test = preprocess_params_transform(SR_data_test, pre_parameters)

        # For signal model, we shift the mass by -3.5 following RANODE workflow
        # copy one set for signal model
        SR_data_test_model_S = SR_data_test.copy()
        # shift mass by -3.5 for signals
        SR_data_test_model_S[:, 0] -= 3.5
        np.save(self.output()["SR_data_test_model_S"].path, SR_data_test_model_S)

        # copy another set for background model
        SR_data_test_model_B = SR_data_test.copy()
        np.save(self.output()["SR_data_test_model_B"].path, SR_data_test_model_B)
