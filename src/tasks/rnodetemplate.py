import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import pickle
import torch
import json

from src.utils.law import BaseTask, SignalNumberMixin, TemplateRandomMixin, TemplateUncertaintyMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder

class RNodeTemplate(
    TemplateRandomMixin,
    SignalNumberMixin,
    BaseTask,
):
    
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=1024)
    epochs = luigi.IntParameter(default=100)
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

        # fixing random seed
        torch.manual_seed(self.train_random_seed)
        
        print("loading data")
        # load data
        data_train_SR_B = np.load(self.input()['preprocessing']['data_train_SR_B'].path)
        data_val_SR_B = np.load(self.input()['preprocessing']['data_val_SR_B'].path)
        # bkg prob predicted by model_B
        data_train_SR_B_prob = np.load(self.input()['bkgprob']['log_B_train'].path)
        data_val_SR_B_prob = np.load(self.input()['bkgprob']['log_B_val'].path)
        # p(m) for bkg model p(x|m)
        SR_mass_hist = np.load(self.input()['preprocessing']['SR_mass_hist'].path)
        SR_mass_bins = np.load(self.input()['preprocessing']['SR_mass_bins'].path)
        density_back = rv_histogram((SR_mass_hist, SR_mass_bins))

        # data to train model_S
        data_train_SR_S = np.load(self.input()['preprocessing']['data_train_SR_S'].path)
        data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
        w_true = (data_train_SR_S[:, -1]==1).sum() / data_train_SR_S.shape[0]

        # convert data to torch tensors
        # log B prob in SR
        log_B_train_tensor = torch.from_numpy(data_train_SR_B_prob.astype('float32')).to(self.device)
        log_B_val_tensor = torch.from_numpy(data_val_SR_B_prob.astype('float32')).to(self.device)
        # bkg in SR
        traintensor_B = torch.from_numpy(data_train_SR_B.astype('float32')).to(self.device)
        valtensor_B = torch.from_numpy(data_val_SR_B.astype('float32')).to(self.device)
        # p(m) for bkg model p(x|m)
        train_mass_prob_B = torch.from_numpy(density_back.pdf(traintensor_B[:,0].cpu().detach().numpy())).to(self.device)
        val_mass_prob_B = torch.from_numpy(density_back.pdf(valtensor_B[:,0].cpu().detach().numpy())).to(self.device)

        # data to train model_S
        traintensor_S = torch.from_numpy(data_train_SR_S.astype('float32')).to(self.device)
        valtensor_S = torch.from_numpy(data_val_SR_S.astype('float32')).to(self.device)
        print("data loaded")
        print("train val data shape: ", traintensor_S.shape, valtensor_S.shape)
        print("w_true: ", w_true)

        # define training input tensors
        train_tensor = torch.utils.data.TensorDataset(traintensor_S, log_B_train_tensor, train_mass_prob_B)
        val_tensor = torch.utils.data.TensorDataset(valtensor_S, log_B_val_tensor, val_mass_prob_B)

        batch_size = self.batchsize
        trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        test_batch_size=batch_size*5
        valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)

        # define model
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from nflow_utils import flows_model_RQS, r_anode_mass_joint_untransformed
        from utils import inverse_sigmoid

        model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)

        w_ = self.w_value
        optimizer = torch.optim.AdamW(model_S.parameters(),lr=3e-4)

        # model scratch
        trainloss_list=[]
        valloss_list=[]
        scrath_path = os.environ.get("SCRATCH_DIR")
        if not os.path.exists(scrath_path + "/model_S/"):
            os.makedirs(scrath_path + "/model_S/")
        else:
            # remove old models
            for file in os.listdir(scrath_path + "/model_S/"):
                os.remove(scrath_path + "/model_S/" + file)


        # define training
        for epoch in range(self.epochs):
            
            # create a dummy params since it is not used in r_anode_mass_joint_untransformed function
            params = {'CR':[], 'SR':[]}

            train_loss = r_anode_mass_joint_untransformed(model_S=model_S,model_B=None,w=w_,optimizer=optimizer,data_loader=trainloader, 
                                                          params=params, device=self.device, mode='train',
                                                          w_train=False)
            val_loss = r_anode_mass_joint_untransformed(model_S=model_S,model_B=None,w=w_,optimizer=optimizer,data_loader=valloader,
                                                        params=params, device=self.device, mode='val',
                                                        w_train=False)



            torch.save(model_S.state_dict(), scrath_path+'/model_S/model_S_'+str(epoch)+'.pt')

            trainloss_list.append(train_loss)
            valloss_list.append(val_loss)
            print('Epoch: ', epoch, 'Train loss: ', train_loss, 'Val loss: ', val_loss)

        # save train and val loss
        trainloss_list=np.array(trainloss_list)
        valloss_list=np.array(valloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)

        # save best models with lowest val loss
        best_models = np.argsort(valloss_list)[:self.num_model_to_save]
        for i in range(self.num_model_to_save):
            print(f'best model {i}: {best_models[i]}, valloss: {valloss_list[best_models[i]]}')
            os.rename(scrath_path+'/model_S/model_S_'+str(best_models[i])+'.pt', self.output()["sig_models"][i].path)

        # save metadata
        metadata = {"w_true": w_true}
        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)


class ScanRANODEoverW(
    TemplateUncertaintyMixin,
    SignalNumberMixin,
    BaseTask,
):
    
    w_min = luigi.FloatParameter(default=0.001)
    w_max = luigi.FloatParameter(default=0.1)
    scan_number = luigi.IntParameter(default=10)
    num_model_to_avg = luigi.IntParameter(default=10)

    def requires(self):
        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

        return model_list
    
    def output(self):
        return {
            "scan_results": self.local_target("scan_results.json"),
            "scan_plot": self.local_target("scan_plot.pdf"),
            "metadata": self.local_target("metadata.json"),
        }
    
    @law.decorator.safe_output
    def run(self):

        results = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        w_true = self.input()["model_0"][0]["metadata"].load()["w_true"]

        for index_w in range(self.scan_number):

            valloss_list = []

            for index_seed in range(self.num_templates):
                trainloss_index = np.load(self.input()[f"model_{index_w}"][index_seed]["trainloss_list"].path)
                valloss_index = np.load(self.input()[f"model_{index_w}"][index_seed]["valloss_list"].path)

                # takr min val losses to calculate the avg and std
                best_models = np.argsort(valloss_index)[:self.num_model_to_avg]
                min_valloss = valloss_index[best_models]

                valloss_list.extend(min_valloss)

            valloss_list = np.array(valloss_list)
            mean_valloss = np.mean(valloss_list)
            std_valloss = np.std(valloss_list)

            results[f"model_{index_w}"] = {
                "w": w_range[index_w],
                "mean_valloss": mean_valloss,
                "std_valloss": std_valloss,
                "valloss_list": valloss_list,
            }

        self.output()["scan_results"].parent.touch()
        with open(self.output()["scan_results"].path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)

        # plot
        import matplotlib.pyplot as plt
        plt.figure()
        w_range_log = np.log10(w_range)
        val_loss = np.array([results[f"model_{index}"]["mean_valloss"] for index in range(self.scan_number)])
        val_loss_std = np.array([results[f"model_{index}"]["std_valloss"] for index in range(self.scan_number)])
        plt.plot(w_range_log, -1 * val_loss, color='r', label='w_scan')
        plt.errorbar(w_range_log, -1 * val_loss, yerr=val_loss_std, fmt='o', color='r')
        
        # plot vertical line at w_true
        plt.axvline(np.log10(w_true), color='b', label='w_true')

        plt.xlabel('log10(w)')
        plt.ylabel('likelihood')
        plt.title(f'w scan likelihood, w_true is {w_true:.5f}')
        plt.legend()
        plt.savefig(self.output()["scan_plot"].path)

        # save metadata
        w_best_index = np.argmin(val_loss)
        w_best = w_range[w_best_index]
        metadata = {"w_true": w_true, "w_best": w_best, "w_best_index": w_best_index}
        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)

        # TODO:
        # train with random seed, show errorbar of 10 models * K seed tests
        # use test set not validation set?
        # do the adjust w fitting, not scanning