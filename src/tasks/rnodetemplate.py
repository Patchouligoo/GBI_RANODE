import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import pickle
import torch

from src.utils.law import BaseTask, SignalNumberMixin, RandomSeedMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb

class RNodeTemplate(
    RandomSeedMixin,
    SignalNumberMixin,
    BaseTask,
):
    
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=1024)
    epochs = luigi.IntParameter(default=200)
    w_value = luigi.FloatParameter(default=0.05)

    def requires(self):
        return {
            'preprocessing': Preprocessing.req(self),
            'bkgprob': PredictBkgProb.req(self),
        }

    def output(self):
        return {
            "sig_model": law.LocalFileTarget("sig_model.pt"),
        }
    
    @law.decorator.safe_output 
    def run(self):
        
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



            torch.save(model_S.state_dict(), scrath_path+'/model_B/model_S_'+str(epoch)+'.pt')

            trainloss_list.append(train_loss)
            valloss_list.append(val_loss)
            print('Epoch: ', epoch, 'Train loss: ', train_loss, 'Val loss: ', val_loss)
