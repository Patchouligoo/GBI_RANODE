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
    SignalNumberMixin,
    BaseTask
):
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=512)
    epochs = luigi.IntParameter(default=200)

    def requires(self):
        return Preprocessing.req(self)
    
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

