import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

from src.utils.law import BaseTask, SignalNumberMixin

class Preprocessing(
    SignalNumberMixin,
    BaseTask
):

    def output(self):
        return {
            "data_train_SR_S": self.local_target("data_train_sr_s.npy"),
            "data_val_SR_S": self.local_target("data_val_sr_s.npy"),
            "data_test_SR_S": self.local_target("data_test_sr_s.npy"),
            "data_train_SR_B": self.local_target("data_train_sr_b.npy"),
            "data_val_SR_B": self.local_target("data_val_sr_b.npy"),
            "data_test_SR_B": self.local_target("data_test_sr_b.npy"),

            "data_train_CR": self.local_target("data_train_cr.npy"),
            "data_val_CR": self.local_target("data_val_cr.npy"),
            
            "SR_mass_hist": self.local_target("SR_mass_hist.npy"),
            "SR_mass_bins": self.local_target("SR_mass_bins.npy"),
            "pre_parameters": self.local_target("pre_parameters.pkl"),
        }

    @law.decorator.safe_output 
    def run(self):
        
        data_dir = os.environ.get("DATA_DIR")
        ranode_path = os.environ.get("RANODE")

        sys.path.append(ranode_path)
        from generate_data_lhc import resample_split
        from utils import logit_transform, preprocess_params_transform, preprocess_params_fit

        SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = self.n_sig, resample_seed = 42,resample = True)

        print('SR shape: ', SR_data.shape)
        print('CR shape: ', CR_data.shape)
        print('true_w: ', true_w)
        print('sigma: ', sigma)

        # ----------------------- calculate normalizing parameters -----------------------
        pre_parameters = preprocess_params_fit(CR_data)
        # save pre_parameters
        self.output()["pre_parameters"].parent.touch()
        with open(self.output()["pre_parameters"].path, 'wb') as f:
            pickle.dump(pre_parameters, f)

        # ----------------------- process data in CR -----------------------
        x_train_CR = preprocess_params_transform(CR_data, pre_parameters)
        data_train_CR, data_val_CR = train_test_split(x_train_CR, test_size=0.1, random_state=42)

        # save training and validation data in CR
        np.save(self.output()["data_train_CR"].path, data_train_CR)
        np.save(self.output()["data_val_CR"].path, data_val_CR)

        # ----------------------- process training data in SR -----------------------
        mass = SR_data[:,0]
        bins = np.linspace(3.3, 3.7, 50)
        hist_back = np.histogram(mass, bins=bins, density=True)
        # save mass histogram and bins
        np.save(self.output()["SR_mass_hist"].path, hist_back[0])
        np.save(self.output()["SR_mass_bins"].path, hist_back[1])

        _, mask = logit_transform(SR_data[:,1:-1], pre_parameters['min'],
                             pre_parameters['max'])
        
        # training data for model_S
        x_train = SR_data[mask]
        x_train = preprocess_params_transform(x_train, pre_parameters) 

        # here x_train will be feed into both model_S and model_B later, to get prob of signal and background


        # ----------------------- load and process testing data in SR -----------------------
        _x_test = np.load(f'{data_dir}/x_test.npy')
        _, mask_test = logit_transform(_x_test[:,1:-1], pre_parameters['min'],
                                pre_parameters['max'])
        x_test = _x_test[mask_test]
        x_test = preprocess_params_transform(x_test, pre_parameters)

        
        
        data_train, data_val = train_test_split(x_train, test_size=0.2, random_state=42)

        # copy one set for signal model
        data_train_s = data_train.copy()
        data_val_s = data_val.copy()
        data_test_s = x_test.copy()
        # shift mass by -3.5 for signals
        data_train_s[:,0] -= 3.5
        data_val_s[:,0] -= 3.5
        data_test_s[:,0] -= 3.5

        np.save(self.output()["data_train_SR_S"].path, data_train_s)
        np.save(self.output()["data_val_SR_S"].path, data_val_s)
        np.save(self.output()["data_test_SR_S"].path, data_test_s)

        # copy another set for background model
        data_train_b = data_train.copy()
        data_val_b = data_val.copy()
        data_test_b = x_test.copy()

        np.save(self.output()["data_train_SR_B"].path, data_train_b)
        np.save(self.output()["data_val_SR_B"].path, data_val_b)
        np.save(self.output()["data_test_SR_B"].path, data_test_b)
