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

from src.utils.law import BaseTask, SignalStrengthMixin, TemplateRandomMixin, TemplateUncertaintyMixin, ProcessMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder

class RNodeTemplate(
    TemplateRandomMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    device = luigi.Parameter(default="cuda:0")
    batchsize = luigi.IntParameter(default=2048)
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
        data_train_SR_B = np.load(self.input()['preprocessing']['data_train_SR_model_B'].path)
        data_val_SR_B = np.load(self.input()['preprocessing']['data_val_SR_model_B'].path)
        # bkg prob predicted by model_B
        data_train_SR_B_prob = np.load(self.input()['bkgprob']['log_B_train'].path)
        data_val_SR_B_prob = np.load(self.input()['bkgprob']['log_B_val'].path)
        # p(m) for bkg model p(x|m)
        with open(self.input()['preprocessing']['SR_mass_hist'].path, 'r') as f:
            mass_hist = json.load(f)
        SR_mass_hist = np.array(mass_hist['hist'])
        SR_mass_bins = np.array(mass_hist['bins'])
        density_back = rv_histogram((SR_mass_hist, SR_mass_bins))

        # data to train model_S
        data_train_SR_S = np.load(self.input()['preprocessing']['data_train_SR_model_S'].path)
        data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_model_S'].path)
        # data to train model_S
        traintensor_S = torch.from_numpy(data_train_SR_S.astype('float32')).to(self.device)
        valtensor_S = torch.from_numpy(data_val_SR_S.astype('float32')).to(self.device)
        
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

        print("data loaded")
        print("train val data shape: ", traintensor_S.shape, valtensor_S.shape)
        print("w_true: ", self.s_ratio)

        # define training input tensors
        train_tensor = torch.utils.data.TensorDataset(traintensor_S, log_B_train_tensor, train_mass_prob_B)
        val_tensor = torch.utils.data.TensorDataset(valtensor_S, log_B_val_tensor, val_mass_prob_B)

        batch_size = self.batchsize
        trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        test_batch_size=batch_size*5
        valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)

        # define model
        from src.models.model_S import r_anode_mass_joint_untransformed, flows_model_RQS

        model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
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

            train_loss = r_anode_mass_joint_untransformed(model_S=model_S,w=self.w_value,optimizer=optimizer,data_loader=trainloader, 
                                                          device=self.device, mode='train')
            val_loss = r_anode_mass_joint_untransformed(model_S=model_S,w=self.w_value,optimizer=optimizer,data_loader=valloader,
                                                        device=self.device, mode='val')

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
        metadata = {"w_true": self.s_ratio, "num_train_events" : traintensor_S.shape[0], "num_val_events": valtensor_S.shape[0]}
        metadata["min_val_loss_list"] = valloss_list[best_models]
        metadata["min_train_loss_list"] = trainloss_list[best_models]

        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)


class ScanRANODEoverW(
    TemplateUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    
    w_min = luigi.FloatParameter(default=0.0001)
    w_max = luigi.FloatParameter(default=0.05)
    scan_number = luigi.IntParameter(default=10)

    def requires(self):
        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

        return model_list
    
    def output(self):
        return self.local_target("fitting_result.pdf")
    
    @law.decorator.safe_output
    def run(self):

        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)
        w_range_log = np.log10(w_range)

        val_loss_scan = []

        for index_w in range(self.scan_number):

            val_loss_list = []

            for index_seed in range(self.num_templates):
                metadata_w_i = self.input()[f"model_{index_w}"][index_seed]["metadata"].load()
                min_val_loss_list = metadata_w_i["min_val_loss_list"]
                val_events_num = metadata_w_i["num_val_events"]
                val_loss_list.extend(min_val_loss_list)

            val_loss_scan.append(val_loss_list)

        # multiple by -1 since the loss is -log[mu*P(sig) + (1-mu)*P(bkg)] but we want likelihood
        # which is log[mu*P(sig) + (1-mu)*P(bkg)]
        val_loss_scan = -1 * np.array(val_loss_scan)

        from src.fitting.fitting import fit_likelihood
        self.output().parent.touch()
        fit_likelihood(w_range_log, val_loss_scan, np.log10(self.s_ratio), val_events_num, self.output().path)


        

        
# class GenerateSignals(
#    ScanRANODEoverW, 
# ):
#     device = luigi.Parameter(default="cuda")
#     n_signal_samples = luigi.IntParameter(default=10000)
    
#     def requires(self):
#         model_list = {}
#         w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

#         for index in range(self.scan_number):
#             model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

#         w_scan_results = ScanRANODEoverW.req(self)

#         return {
#             "models": model_list,
#             "w_scan_results": w_scan_results,
#         } 

#     def output(self):
#         return {
#             "signal_list": self.local_target("signal_list.npy"),
#         }

#     @law.decorator.safe_output
#     def run(self):

#         # load previous scan result
#         w_scan_results = self.input()["w_scan_results"]["metadata"].load()
#         w_best = w_scan_results["w_best"]
#         w_best_index = w_scan_results["w_best_index"]

#         # load the model at best w
#         model_best_list = []
#         for rand_seed_index in range(len(self.input()["models"][f"model_{w_best_index}"])):
#             model_best_seed_i = self.input()["models"][f"model_{w_best_index}"][rand_seed_index]["sig_models"]
#             for model in model_best_seed_i:
#                 model_best_list.append(model.path)

#         # define model
#         ranode_path = os.environ.get("RANODE")
#         sys.path.append(ranode_path)
#         from nflow_utils import flows_model_RQS

#         # generate signals from each models
#         signal_list = []
#         for model_dir in model_best_list:
#             model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
#             model_S.load_state_dict(torch.load(model_dir))
#             model_S.eval()

#             signal_samples = model_S.sample(self.n_signal_samples)
#             signal_list.append(signal_samples.cpu().detach().numpy())

#             # clean cuda memory
#             del model_S
#             torch.cuda.empty_cache()

#         # sample weight is 1 / num_models
#         signal_list = np.array(signal_list)
#         sample_weight = 1 / signal_list.shape[0]
#         sample_weight = np.ones((len(signal_list), self.n_signal_samples, 1)) * sample_weight

#         signal_list = np.concatenate([signal_list, sample_weight], axis=-1)

#         signal_list = signal_list.reshape(-1, 6)

#         self.output()["signal_list"].parent.touch()
#         np.save(self.output()["signal_list"].path, signal_list)


# class SignalGenerationPlot(
#     GenerateSignals,
# ):
#     nbins = luigi.IntParameter(default=41)

#     def requires(self):
#         return {
#             "generated_signal_list": GenerateSignals.req(self, n_signal_samples=self.n_sig),
#             "preprocessing": Preprocessing.req(self),
#         }

#     def output(self):
#         return self.local_target("signal_plot.pdf")

#     @law.decorator.safe_output
#     def run(self):

#         generated_signals = np.load(self.input()["generated_signal_list"]["signal_list"].path)
#         generated_signal_features = generated_signals[:, 1:-1]
#         generated_signal_weights = generated_signals[:, -1]
#         generated_signals_mass = generated_signal_features[:, 0]

#         # load data
#         data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
#         mask_signals_val = data_val_SR_S[:, -1] == 1
#         signal_val = data_val_SR_S[mask_signals_val]
#         signal_val_features = signal_val[:, 1:-1]
#         signal_val_mass = signal_val_features[:, 0]

#         mask_bkg_val = data_val_SR_S[:, -1] == 0
#         bkg_val = data_val_SR_S[mask_bkg_val]
#         bkg_val_features = bkg_val[:, 1:-1]
#         bkg_val_mass = bkg_val_features[:, 0]

#         # plot
#         import matplotlib.pyplot as plt
#         from matplotlib.backends.backend_pdf import PdfPages

#         self.output().parent.touch()
#         with PdfPages(self.output().path) as pdf:
            
#             for feature_index in range(signal_val_features.shape[1]):

#                 bins = np.linspace(bkg_val_features[:, feature_index].min(), bkg_val_features[:, feature_index].max(), self.nbins)
                                   
#                 f = plt.figure()
#                 plt.hist(signal_val_features[:, feature_index], bins=bins, alpha=0.5, label='val signal', density=True, histtype='step', lw=3)
#                 plt.hist(bkg_val_features[:, feature_index], bins=bins, alpha=0.5, label='val bkg', density=True, histtype='step', lw=3)
#                 plt.hist(generated_signal_features[:, feature_index], bins=bins, weights=generated_signal_weights, 
#                          alpha=0.5, label='generated signal', density=True, histtype='step', lw=3)
#                 plt.xlabel(f'feature {feature_index}')
#                 plt.ylabel('density')
#                 plt.title(f'feature {feature_index} distribution, {self.n_sig} signal in samples')
#                 plt.legend()
#                 plt.yscale('log')
#                 pdf.savefig(f)
#                 plt.close(f)



