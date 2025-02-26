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

from src.utils.law import BaseTask, SignalStrengthMixin, ProcessMixin, TemplateRandomMixin, BkginSRDataMixin


class ProcessSignal(
    ProcessMixin,
    BaseTask
):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=1)
    """

    def output(self):
        return self.local_target("reprocessed_signals.npy")

    @law.decorator.safe_output 
    def run(self):
        data_dir = os.environ.get("DATA_DIR")
        data_path = f'{data_dir}/extra_raw_lhco_samples/events_anomalydetection_Z_XY_qq_parametric.h5'

        from src.data_prep.signal_processing import process_signals

        self.output().parent.touch()
        process_signals(data_path, self.output().path, self.mx, self.my)


class ProcessBkg(
    BaseTask
):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=0)
    """

    # bkg_type = luigi.ChoiceParameter(choices=["qcd", "extra_qcd"], default="qcd")

    # def store_parts(self):
    #     return super().store_parts() + (self.bkg_type,)

    def output(self):
        return self.local_target("reprocessed_bkgs.npy")

    @law.decorator.safe_output 
    def run(self):
        data_dir = os.environ.get("DATA_DIR")

        data_path_qcd = f'{data_dir}/events_anomalydetection_v2.features.h5'
        data_path_extra_qcd = f'{data_dir}/events_anomalydetection_qcd_extra_inneronly_features.h5'

        from src.data_prep.bkg_processing import process_bkgs
        
        output_qcd = process_bkgs(data_path_qcd)
        output_extra_qcd = process_bkgs(data_path_extra_qcd)

        output_combined = np.concatenate([output_qcd, output_extra_qcd], axis=0)

        self.output().parent.touch()
        np.save(self.output().path, output_combined)


class Preprocessing(
    ProcessMixin,
    SignalStrengthMixin,
    BkginSRDataMixin,
    BaseTask
):
    
    def requires(self):
        return {
            "signal": ProcessSignal.req(self),
            "bkg": ProcessBkg.req(self),
        }

    def output(self):
        return {
            "SR_data_trainval_model_S": self.local_target("data_SR_data_trainval_model_S.npy"),
            "data_test_SR_model_S": self.local_target("data_test_sr_s.npy"),

            "SR_data_trainval_model_B": self.local_target("data_SR_data_trainval_model_B.npy"),
            "data_test_SR_model_B": self.local_target("data_test_sr_b.npy"),

            "data_train_CR": self.local_target("data_train_cr.npy"),
            "data_val_CR": self.local_target("data_val_cr.npy"),
            
            "SR_mass_hist": self.local_target("SR_mass_hist.json"),
            "pre_parameters": self.local_target("pre_parameters.json"),
        }

    @law.decorator.safe_output 
    def run(self):
        
        import json
        from src.data_prep.data_prep import sample_split #, resample_split_test
        from src.data_prep.utils import logit_transform, preprocess_params_transform, preprocess_params_fit

        signal_path = self.input()["signal"].path
        bkg_path = self.input()["bkg"].path

        SR_data_trainval, SR_data_test, CR_data = sample_split(signal_path, bkg_path, sig_ratio = self.s_ratio, bkg_num_in_sr_data=self.bkg_num_in_sr_data, resample_seed = 42)
        # SR_data_test = resample_split_test(signal_path, bkg_path_test, resample_seed = 42)

        # print('true_w in data: ', true_w)
        # print('design w in data: ', self.s_ratio)
        
        # ----------------------- calculate normalizing parameters -----------------------
        pre_parameters = preprocess_params_fit(CR_data)
        # save pre_parameters
        self.output()["pre_parameters"].parent.touch()
        with open(self.output()["pre_parameters"].path, 'w') as f:
            json.dump(pre_parameters, f, cls=NumpyEncoder)

        # # ----------------------- process data in CR -----------------------
        CR_data = preprocess_params_transform(CR_data, pre_parameters)
        CR_data_train, CR_data_val = train_test_split(CR_data, test_size=0.25, random_state=42)

        # save training and validation data in CR
        np.save(self.output()["data_train_CR"].path, CR_data_train)
        np.save(self.output()["data_val_CR"].path, CR_data_val)

        # ----------------------- process data in SR -----------------------
        from config.configs import SR_MIN, SR_MAX
        mass = SR_data_trainval[SR_data_trainval[:,-1]==0,0]
        bins = np.linspace(SR_MIN, SR_MAX, 50)
        hist_back = np.histogram(mass, bins=bins, density=True)
        # save mass histogram and bins
        with open(self.output()["SR_mass_hist"].path , 'w') as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # SR_data_trainval
        _, mask = logit_transform(SR_data_trainval[:,1:-1], pre_parameters['min'],
                             pre_parameters['max'])
        SR_data_trainval = SR_data_trainval[mask]
        SR_data_trainval = preprocess_params_transform(SR_data_trainval, pre_parameters) 
        # here x_train will be feed into both model_S and model_B later, to get prob of signal and background

        # testing data
        _, mask = logit_transform(SR_data_test[:,1:-1], pre_parameters['min'],
                             pre_parameters['max'])
        SR_data_test = SR_data_test[mask]
        SR_data_test = preprocess_params_transform(SR_data_test, pre_parameters)
        
        # For signal model, we shift the mass by -3.5 following RANODE workflow
        # copy one set for signal model
        SR_data_trainval_model_S = SR_data_trainval.copy()
        SR_data_test_model_S = SR_data_test.copy()
        # shift mass by -3.5 for signals
        SR_data_trainval_model_S[:,0] -= 3.5
        SR_data_test_model_S[:,0] -= 3.5

        np.save(self.output()["SR_data_trainval_model_S"].path, SR_data_trainval_model_S)
        np.save(self.output()["data_test_SR_model_S"].path, SR_data_test_model_S)

        # copy another set for background model
        SR_data_trainval_model_B = SR_data_trainval.copy()
        SR_data_test_model_B = SR_data_test.copy()

        np.save(self.output()["SR_data_trainval_model_B"].path, SR_data_trainval_model_B)
        np.save(self.output()["data_test_SR_model_B"].path, SR_data_test_model_B)
