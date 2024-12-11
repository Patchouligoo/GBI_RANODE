import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import pickle
import torch

from src.utils.law import BaseTask, SignalNumberMixin
from src.tasks.preprocessing import Preprocessing

class BkgTemplateTraining(
    BaseTask
):
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=512)
    epochs = luigi.IntParameter(default=200)

    def requires(self):
        return Preprocessing.req(self, n_sig=0)
    
    def output(self):
        return {
            "bkg_models": [self.local_target("model_CR_"+str(i)+".pt") for i in range(10)],
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
        }
    
    @law.decorator.safe_output 
    def run(self):

        # need:
        # "data_train_CR": self.local_target("data_train_cr.npy"),
        # "data_val_CR": self.local_target("data_val_cr.npy"),

        # load data
        data_train_CR = np.load(self.input()["data_train_CR"].path)
        data_val_CR = np.load(self.input()["data_val_CR"].path)

        traintensor = torch.from_numpy(data_train_CR.astype('float32')).to(self.device)
        valtensor = torch.from_numpy(data_val_CR.astype('float32')).to(self.device)

        train_tensor = torch.utils.data.TensorDataset(traintensor)
        val_tensor = torch.utils.data.TensorDataset(valtensor)

        trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=self.batchsize, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_tensor, batch_size=self.batchsize*5, shuffle=False)

        # define model 
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from density_estimator import DensityEstimator
        config_file = os.path.join(os.path.dirname(ranode_path), "scripts", "DE_MAF_model.yml")
        
        model_B = DensityEstimator(config_file, eval_mode=False, device=self.device)
        
        # define training
        from nflow_utils import anode

        trainloss_list=[]
        valloss_list=[]
        scrath_path = os.environ.get("SCRATCH_DIR")
        if not os.path.exists(scrath_path + "/model_B/"):
            os.makedirs(scrath_path + "/model_B/")
        else:
            # remove old models
            for file in os.listdir(scrath_path + "/model_B/"):
                os.remove(scrath_path + "/model_B/" + file)

        # no early stopping, just run 200 epochs and take 10 lowest valloss models
        for epoch in range(self.epochs):

            trainloss=anode(model_B.model,trainloader,model_B.optimizer,params=None ,device=self.device, mode='train')
            valloss=anode(model_B.model,valloader,model_B.optimizer,params=None, device=self.device, mode='val')

            torch.save(model_B.model.state_dict(), scrath_path+'/model_B/model_CR_'+str(epoch)+'.pt')

            valloss_list.append(valloss)
            trainloss_list.append(trainloss)

            print('epoch: ', epoch, 'trainloss: ', trainloss, 'valloss: ', valloss)


        # save trainings and validation losses
        trainloss_list=np.array(trainloss_list)
        valloss_list=np.array(valloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)

        # save 10 best models
        best_models = np.argsort(valloss_list)[:10]
        for i in range(10):
            print(f'best model {i}: {best_models[i]}, valloss: {valloss_list[best_models[i]]}')
            os.rename(scrath_path+'/model_B/model_CR_'+str(best_models[i])+'.pt', self.output()["bkg_models"][i].path)


class BkgTemplateChecking(
    SignalNumberMixin,
    BaseTask, 
):
    
    device = luigi.Parameter(default="cuda")
    num_CR_samples = luigi.IntParameter(default=100000)
    
    def requires(self):
        return {
            "bkg_models": BkgTemplateTraining.req(self),
            "preprocessed_data": Preprocessing.req(self),
        }
    
    def output(self):
        return {
            "loss_plot": self.local_target("loss_plot.pdf"),
            "CR_comparison_plot": self.local_target("CR_comparison_plots.pdf"),
            "SR_comparison_plot": self.local_target("SR_comparison_plots.pdf"),
        }
    
    @law.decorator.safe_output
    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # ----------------------------------- plot loss -----------------------------------
        self.output()['loss_plot'].parent.touch()
        train_loss = np.load(self.input()["bkg_models"]["trainloss_list"].path)
        val_loss = np.load(self.input()["bkg_models"]["valloss_list"].path)
        f = plt.figure()
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        f.savefig(self.output()["loss_plot"].path)

        # ----------------------------------- load model --------------------------------
        # define model 
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        # from utils import inverse_transform
        # load the models
        from density_estimator import DensityEstimator
        config_file = os.path.join(os.path.dirname(ranode_path), "scripts", "DE_MAF_model.yml")
        model_B = DensityEstimator(config_file, eval_mode=True, device=self.device)
        best_model_dir = self.input()["bkg_models"]["bkg_models"][0].path
        model_B.model.load_state_dict(torch.load(best_model_dir))
        model_B.model.to(self.device)
        model_B.model.eval()

        # -------------------------------- CR comparison plots --------------------------------
        # load the sample to compare with
        data_train_CR = np.load(self.input()["preprocessed_data"]["data_train_CR"].path)

        # generate CR events using the model with condition from data_train_CR
        mass_cond_CR = torch.from_numpy(data_train_CR[:,0]).reshape((-1, 1)).type(torch.FloatTensor).to(self.device)
        mass_cond_CR = mass_cond_CR[:self.num_CR_samples]

        with torch.no_grad():
            sampled_CR_events = model_B.model.sample(num_samples=len(mass_cond_CR), cond_inputs=mass_cond_CR)

        sampled_CR_events = sampled_CR_events.cpu().numpy().astype('float32')

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
                bins = np.linspace(data_train_CR[:,i+1].min(), data_train_CR[:,i+1].max(), 100)
                f = plt.figure()
                plt.hist(data_train_CR[:self.num_CR_samples,i+1], bins=bins, histtype='step', label='data_train_CR')
                plt.hist(sampled_CR_events[:,i], bins=bins, histtype='step', label='sampled CR events')
                plt.xlabel(f'var {i}')
                plt.ylabel('counts')
                plt.legend()
                pdf.savefig(f)
                plt.close(f)

        # -------------------------------- SR comparison plots --------------------------------
        # we load 10 model_Bs
        model_Bs = []
        for i in range(10):
            model_B = DensityEstimator(config_file, eval_mode=True, device=self.device)
            best_model_dir = self.input()["bkg_models"]["bkg_models"][i].path
            model_B.model.load_state_dict(torch.load(best_model_dir))
            model_B.model.to(self.device)
            model_B.model.eval()
            model_Bs.append(model_B)


        # load the sample to compare with
        data_train_SR_B = np.load(self.input()["preprocessed_data"]["data_train_SR_B"].path) # with background only
        SR_mass_hist = np.load(self.input()["preprocessed_data"]["SR_mass_hist"].path)
        SR_mass_bins = np.load(self.input()["preprocessed_data"]["SR_mass_bins"].path)

        # generate SR events using the model with condition from data_train_SR_B
        mass_cond_SR = data_train_SR_B[:,0]
        density_hist = rv_histogram((SR_mass_hist, SR_mass_bins))
        
        uniform_mass_SR = np.linspace(SR_mass_bins.min(), SR_mass_bins.max(), len(mass_cond_SR))
        uniform_mass_SR_weights = density_hist.pdf(uniform_mass_SR)
        uniform_mass_SR_weights = uniform_mass_SR_weights / uniform_mass_SR_weights.sum() * len(mass_cond_SR)

        uniform_mass_SR = torch.from_numpy(uniform_mass_SR).reshape((-1, 1)).type(torch.FloatTensor).to(self.device)

        # sample from all model_Bs, and average the results in histograms
        sampled_SR_events_list = []
        sampled_SR_events_list_weight = []
        for model_B in model_Bs:
            with torch.no_grad():
                sampled_SR_events = model_B.model.sample(num_samples=len(uniform_mass_SR), cond_inputs=uniform_mass_SR)
                sampled_SR_events = sampled_SR_events.cpu().numpy().astype('float32')
                sampled_SR_events_list.append(sampled_SR_events)
                sampled_SR_events_list_weight.append(uniform_mass_SR_weights)

        sampled_SR_events = np.array(sampled_SR_events_list).reshape(-1, len(sampled_SR_events[0]))
        sampled_SR_events_weight = np.array(sampled_SR_events_list_weight).flatten() / len(model_Bs)

        uniform_mass_SR = uniform_mass_SR.cpu().numpy().reshape(-1).astype('float32')

        # plot the comparison
        with PdfPages(self.output()["SR_comparison_plot"].path) as pdf:
            # first plot the mass distribution
            f = plt.figure()
            plt.hist(mass_cond_SR, bins=SR_mass_bins, histtype='step', label='mass condition SR')
            plt.hist(uniform_mass_SR, bins=SR_mass_bins, weights=uniform_mass_SR_weights, histtype='step', label='uniform mass SR')
            plt.xlabel('mass')
            plt.ylabel('counts')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            # then plot the rest of the variables
            for i in range(len(sampled_SR_events[0])):
                bins = np.linspace(data_train_SR_B[:,i+1].min(), data_train_SR_B[:,i+1].max(), 100)
                f = plt.figure()
                plt.hist(sampled_SR_events[:,i], bins=bins, weights=sampled_SR_events_weight, histtype='step', label='sampled SR events')
                plt.hist(data_train_SR_B[data_train_SR_B[:, -1]==0, i+1], bins=bins, histtype='step', label='data_train_SR (background)')
                plt.hist(data_train_SR_B[data_train_SR_B[:, -1]==1, i+1], bins=bins, histtype='step', label='data_train_SR (signal)')
                plt.hist(data_train_SR_B[:,i+1], bins=bins, histtype='step', label='data_train_SR (all)')
                plt.xlabel(f'var {i}')
                plt.ylabel('counts')
                plt.legend()
                pdf.savefig(f)
                plt.close(f)


class PredictBkgProb(
    SignalNumberMixin,
    BaseTask, 
):
    
    device = luigi.Parameter(default="cuda")

    def requires(self):
        return {
            "bkg_models": BkgTemplateTraining.req(self),
            "preprocessed_data": Preprocessing.req(self),
        }    
    
    def output(self):
        return {
            "log_B_train": self.local_target("log_B_train.npy"),
            "log_B_val": self.local_target("log_B_val.npy"),
            "log_B_test": self.local_target("log_B_test.npy"),
        }
    
    @law.decorator.safe_output
    def run(self):
        # load the models
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from density_estimator import DensityEstimator
        config_file = os.path.join(os.path.dirname(ranode_path), "scripts", "DE_MAF_model.yml")
        model_Bs = []
        for i in range(10):
            model_B = DensityEstimator(config_file, eval_mode=True, device="cuda")
            best_model_dir = self.input()["bkg_models"]["bkg_models"][i].path
            model_B.model.load_state_dict(torch.load(best_model_dir))
            model_B.model.to("cuda")
            model_B.model.eval()
            model_Bs.append(model_B)

        # load the sample to compare with
        data_train_SR_B = np.load(self.input()["preprocessed_data"]["data_train_SR_B"].path)
        traintensor_SR_B = torch.from_numpy(data_train_SR_B.astype('float32')).to(self.device)
        data_val_SR_B = np.load(self.input()["preprocessed_data"]["data_val_SR_B"].path)
        valtensor_SR_B = torch.from_numpy(data_val_SR_B.astype('float32')).to(self.device)
        data_test_SR_B = np.load(self.input()["preprocessed_data"]["data_test_SR_B"].path)
        testtensor_SR_B = torch.from_numpy(data_test_SR_B.astype('float32')).to(self.device)

        # get avg probility of 10 models
        log_B_train_list = []
        log_B_val_list = []
        log_B_test_list = []
        for model_B in model_Bs:
            with torch.no_grad():
                log_B_train = model_B.model.log_probs(inputs=traintensor_SR_B[:,1:-1], cond_inputs=traintensor_SR_B[:,0].reshape(-1,1))
                log_B_train_list.append(log_B_train.cpu().numpy())
                log_B_val = model_B.model.log_probs(inputs=valtensor_SR_B[:,1:-1], cond_inputs=valtensor_SR_B[:,0].reshape(-1,1))
                log_B_val_list.append(log_B_val.cpu().numpy())
                log_B_test = model_B.model.log_probs(inputs=testtensor_SR_B[:,1:-1], cond_inputs=testtensor_SR_B[:,0].reshape(-1,1))
                log_B_test_list.append(log_B_test.cpu().numpy())

        log_B_train = np.array(log_B_train_list)
        B_train = np.exp(log_B_train).mean(axis=0)
        log_B_train = np.log(B_train + 1e-32)

        log_B_val = np.array(log_B_val_list)
        B_val = np.exp(log_B_val).mean(axis=0)
        log_B_val = np.log(B_val + 1e-32)

        log_B_test = np.array(log_B_test_list)
        B_test = np.exp(log_B_test).mean(axis=0)
        log_B_test = np.log(B_test + 1e-32)
 
        self.output()["log_B_train"].parent.touch()
        np.save(self.output()["log_B_train"].path, log_B_train)
        np.save(self.output()["log_B_val"].path, log_B_val)
        np.save(self.output()["log_B_test"].path, log_B_test)

