import os, sys
import importlib
import luigi
import law
import json
import copy
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import pickle
import torch

from src.utils.law import BaseTask, SignalStrengthMixin, TemplateRandomMixin, BkgTemplateUncertaintyMixin, TranvalSplitRandomMixin, ProcessMixin
from src.tasks.preprocessing import PreprocessingTrainval, ProcessBkg

class BkgTemplateTraining(
    TemplateRandomMixin,
    ProcessMixin,
    BaseTask
):
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=2048)
    epochs = luigi.IntParameter(default=200)
    num_model_to_save = luigi.IntParameter(default=10)

    def requires(self):
        return ProcessBkg.req(self)
    
    def output(self):
        return {
            "bkg_models": [self.local_target("model_CR_"+str(i)+".pt") for i in range(self.num_model_to_save)],
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
        }
    
    @law.decorator.safe_output 
    def run(self):
        
        # freeze the random seed of torch
        torch.manual_seed(self.train_random_seed)

        # need:
        # "data_train_CR": self.local_target("data_train_cr.npy"),
        # "data_val_CR": self.local_target("data_val_cr.npy"),

        # load data
        data_train_CR = np.load(self.input()["CR_train"].path)
        data_val_CR = np.load(self.input()["CR_val"].path)

        traintensor = torch.from_numpy(data_train_CR.astype('float32')).to(self.device)
        valtensor = torch.from_numpy(data_val_CR.astype('float32')).to(self.device)

        train_tensor = torch.utils.data.TensorDataset(traintensor)
        val_tensor = torch.utils.data.TensorDataset(valtensor)

        trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=self.batchsize, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_tensor, batch_size=self.batchsize*5, shuffle=False)

        # define model 
        from src.models.model_B import DensityEstimator, anode
        config_file = os.path.join("src", "models", "DE_MAF_model.yml")
        
        model_B = DensityEstimator(config_file, eval_mode=False, device=self.device)
        
        trainloss_list=[]
        valloss_list=[]
        model_list = []

        # no early stopping, just run 200 epochs and take num_model_to_save lowest valloss models
        for epoch in range(self.epochs):

            trainloss=anode(model_B.model,trainloader,model_B.optimizer,params=None ,device=self.device, mode='train')
            valloss=anode(model_B.model,valloader,model_B.optimizer,params=None, device=self.device, mode='val')

            # torch.save(model_B.model.state_dict(), scrath_path+'/model_B/model_CR_'+str(epoch)+'.pt')
            state_dict = copy.deepcopy({k: v.cpu() for k, v in model_B.model.state_dict().items()})
            model_list.append(state_dict)

            valloss_list.append(valloss)
            trainloss_list.append(trainloss)

            print('epoch: ', epoch, 'trainloss: ', trainloss, 'valloss: ', valloss)


        # save trainings and validation losses
        trainloss_list=np.array(trainloss_list)
        valloss_list=np.array(valloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)

        # save best models
        best_models = np.argsort(valloss_list)
        for i in range(self.num_model_to_save):
            print(f'best model {i}: {best_models[i]}, valloss: {valloss_list[best_models[i]]}')
            torch.save(model_list[best_models[i]], self.output()["bkg_models"][i].path)


class BkgTemplateChecking(
    BkgTemplateUncertaintyMixin,
    ProcessMixin,
    BaseTask, 
):
    
    device = luigi.Parameter(default="cuda")
    num_CR_samples = luigi.IntParameter(default=100000)
    num_model_to_save = luigi.IntParameter(default=10)
    
    def requires(self):
        return {
            "bkg_models": [BkgTemplateTraining.req(self, train_random_seed=i) for i in range(self.num_bkg_templates)],
            "preprocessed_data": ProcessBkg.req(self)
        }
    
    def output(self):
        return {
            "CR_comparison_plot": self.local_target("CR_comparison_plots.pdf"),
        }
    
    @law.decorator.safe_output
    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        self.output()['CR_comparison_plot'].parent.touch()        
        # -------------------------------- CR comparison plots --------------------------------
        # load the sample to compare with
        data_val_CR = np.load(self.input()["preprocessed_data"]["CR_val"].path)
        # generate CR events using the model with condition from data_train_CR
        mass_cond_CR = torch.from_numpy(data_val_CR[:,0]).reshape((-1, 1)).type(torch.FloatTensor).to(self.device)
        mass_cond_CR = mass_cond_CR[:self.num_CR_samples]

        sampled_CR_events = [] 

        # ----------------------------------- load all models and make prediction --------------------------------
        from src.models.model_B import DensityEstimator
        config_file = os.path.join("src", "models", "DE_MAF_model.yml")
        
        for seed_i in range(self.num_bkg_templates):
            for model_epoch_j in range(self.num_model_to_save):
                # load the models
                model_B_seed_i_epoch_j = DensityEstimator(config_file, eval_mode=True, device=self.device)
                best_model_dir_seed_i_epoch_j = self.input()["bkg_models"][seed_i]["bkg_models"][model_epoch_j].path
                model_B_seed_i_epoch_j.model.load_state_dict(torch.load(best_model_dir_seed_i_epoch_j))
                model_B_seed_i_epoch_j.model.to(self.device)
                model_B_seed_i_epoch_j.model.eval()

                with torch.no_grad():
                    sampled_CR_events_seed_i_epoch_j = model_B_seed_i_epoch_j.model.sample(num_samples=len(mass_cond_CR), cond_inputs=mass_cond_CR)

                sampled_CR_events.extend(sampled_CR_events_seed_i_epoch_j.cpu().numpy().astype('float32'))

        sampled_CR_events = np.array(sampled_CR_events)
        sampled_CR_events_weight = np.ones(len(sampled_CR_events)) / len(sampled_CR_events) * self.num_CR_samples

        # plot the comparison
        with PdfPages(self.output()["CR_comparison_plot"].path) as pdf:
            # first plot the mass distribution
            mass_cond_CR = mass_cond_CR.cpu().numpy().reshape(-1).astype('float32')
            f = plt.figure()
            plt.hist(mass_cond_CR, bins=100, histtype='step', label='mass condition CR')
            plt.xlabel('mass')
            plt.ylabel('counts')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            # then plot the rest of the variables
            for i in range(len(sampled_CR_events[0])):
                bins = np.linspace(data_val_CR[:,i+1].min(), data_val_CR[:,i+1].max(), 100)
                f = plt.figure()
                plt.hist(data_val_CR[:self.num_CR_samples,i+1], bins=bins, histtype='step', label='data_train_CR')
                plt.hist(sampled_CR_events[:,i], weights=sampled_CR_events_weight, bins=bins, histtype='step', label='sampled CR events')
                plt.xlabel(f'var {i}')
                plt.ylabel('counts')
                plt.legend()
                pdf.savefig(f)
                plt.close(f)


class PredictBkgProbTrainVal(
    BkgTemplateUncertaintyMixin,
    TranvalSplitRandomMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask
):
    
    num_model_to_save = luigi.IntParameter(default=10)
    device = luigi.Parameter(default="cuda")

    def requires(self):
        return {
            "bkg_models": [BkgTemplateTraining.req(self, train_random_seed=i) for i in range(self.num_bkg_templates)],
            "preprocessed_data": PreprocessingTrainval.req(self, trainval_split_seed=self.trainval_split_seed),
        }
    
    def output(self):
        return {
            "log_B_train": self.local_target("log_B_train.npy"),
            "log_B_val": self.local_target("log_B_val.npy"),
        }
    
    @law.decorator.safe_output
    def run(self):
        # load the models
        from src.models.model_B import DensityEstimator, anode
        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_Bs = []

        for i in range(self.num_bkg_templates):
            for j in range(self.num_model_to_save):
                model_B = DensityEstimator(config_file, eval_mode=True, device="cuda")
                best_model_dir = self.input()["bkg_models"][i]["bkg_models"][j].path
                model_B.model.load_state_dict(torch.load(best_model_dir))
                model_B.model.to("cuda")
                model_B.model.eval()
                model_Bs.append(model_B)

        # load the sample to compare with
        data_train_SR_B = np.load(self.input()["preprocessed_data"]["SR_data_train_model_B"].path)
        traintensor_SR_B = torch.from_numpy(data_train_SR_B.astype('float32')).to(self.device)
        
        data_val_SR_B = np.load(self.input()["preprocessed_data"]["SR_data_val_model_B"].path)
        valtensor_SR_B = torch.from_numpy(data_val_SR_B.astype('float32')).to(self.device)

        # get avg probility of 10 models
        log_B_train_list = []
        log_B_val_list = []
        for model_B in model_Bs:
            with torch.no_grad():
                log_B_train = model_B.model.log_probs(inputs=traintensor_SR_B[:,1:-1], cond_inputs=traintensor_SR_B[:,0].reshape(-1,1))
                # set all nans to 0
                log_B_train[torch.isnan(log_B_train)] = 0
                log_B_train_list.append(log_B_train.cpu().numpy())

                log_B_val = model_B.model.log_probs(inputs=valtensor_SR_B[:,1:-1], cond_inputs=valtensor_SR_B[:,0].reshape(-1,1))
                # set all nans to 0
                log_B_val[torch.isnan(log_B_val)] = 0
                log_B_val_list.append(log_B_val.cpu().numpy())

        log_B_train = np.array(log_B_train_list)
        B_train = np.exp(log_B_train).mean(axis=0)
        log_B_train = np.log(B_train + 1e-32)

        log_B_val = np.array(log_B_val_list)
        B_val = np.exp(log_B_val).mean(axis=0)
        log_B_val = np.log(B_val + 1e-32)
 
        self.output()["log_B_train"].parent.touch()
        np.save(self.output()["log_B_train"].path, log_B_train)
        np.save(self.output()["log_B_val"].path, log_B_val)

