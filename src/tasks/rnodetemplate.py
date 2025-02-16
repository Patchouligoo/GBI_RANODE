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

from src.utils.law import BaseTask, SignalStrengthMixin, TemplateRandomMixin, TemplateUncertaintyMixin
from src.tasks.preprocessing import Preprocessing
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder

class RNodeTemplate(
    TemplateRandomMixin,
    SignalStrengthMixin,
    BaseTask,
):
    
    device = luigi.Parameter(default="cuda:0")
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
        data_test_SR_B = np.load(self.input()['preprocessing']['data_test_SR_B'].path)
        # bkg prob predicted by model_B
        data_train_SR_B_prob = np.load(self.input()['bkgprob']['log_B_train'].path)
        data_val_SR_B_prob = np.load(self.input()['bkgprob']['log_B_val'].path)
        data_test_SR_B_prob = np.load(self.input()['bkgprob']['log_B_test'].path)
        # p(m) for bkg model p(x|m)
        SR_mass_hist = np.load(self.input()['preprocessing']['SR_mass_hist'].path)
        SR_mass_bins = np.load(self.input()['preprocessing']['SR_mass_bins'].path)
        density_back = rv_histogram((SR_mass_hist, SR_mass_bins))

        # data to train model_S
        data_train_SR_S = np.load(self.input()['preprocessing']['data_train_SR_S'].path)
        data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
        data_test_SR_S = np.load(self.input()['preprocessing']['data_test_SR_S'].path)
        
        # convert data to torch tensors
        # log B prob in SR
        log_B_train_tensor = torch.from_numpy(data_train_SR_B_prob.astype('float32')).to(self.device)
        log_B_val_tensor = torch.from_numpy(data_val_SR_B_prob.astype('float32')).to(self.device)
        log_B_test_tensor = torch.from_numpy(data_test_SR_B_prob.astype('float32')).to(self.device)
        # bkg in SR
        traintensor_B = torch.from_numpy(data_train_SR_B.astype('float32')).to(self.device)
        valtensor_B = torch.from_numpy(data_val_SR_B.astype('float32')).to(self.device)
        testtensor_B = torch.from_numpy(data_test_SR_B.astype('float32')).to(self.device)
        # p(m) for bkg model p(x|m)
        train_mass_prob_B = torch.from_numpy(density_back.pdf(traintensor_B[:,0].cpu().detach().numpy())).to(self.device)
        val_mass_prob_B = torch.from_numpy(density_back.pdf(valtensor_B[:,0].cpu().detach().numpy())).to(self.device)
        test_mass_prob_B = torch.from_numpy(density_back.pdf(testtensor_B[:,0].cpu().detach().numpy())).to(self.device)

        # data to train model_S
        traintensor_S = torch.from_numpy(data_train_SR_S.astype('float32')).to(self.device)
        valtensor_S = torch.from_numpy(data_val_SR_S.astype('float32')).to(self.device)
        testtensor_S = torch.from_numpy(data_test_SR_S.astype('float32')).to(self.device)
        print("data loaded")
        print("train val test data shape: ", traintensor_S.shape, valtensor_S.shape, testtensor_S.shape)
        print("w_true: ", self.s_ratio)

        # define training input tensors
        train_tensor = torch.utils.data.TensorDataset(traintensor_S, log_B_train_tensor, train_mass_prob_B)
        val_tensor = torch.utils.data.TensorDataset(valtensor_S, log_B_val_tensor, val_mass_prob_B)
        test_tensor = torch.utils.data.TensorDataset(testtensor_S, log_B_test_tensor, test_mass_prob_B)

        batch_size = self.batchsize
        trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        test_batch_size=batch_size*5
        valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_tensor, batch_size=test_batch_size, shuffle=False)

        # define model
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from nflow_utils import flows_model_RQS, r_anode_mass_joint_untransformed
        from utils import inverse_sigmoid

        model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
        optimizer = torch.optim.AdamW(model_S.parameters(),lr=3e-4)

        # model scratch
        trainloss_list=[]
        valloss_list=[]
        testloss_list=[]
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

            train_loss = r_anode_mass_joint_untransformed(model_S=model_S,model_B=None,w=self.w_value,optimizer=optimizer,data_loader=trainloader, 
                                                          params=params, device=self.device, mode='train',
                                                          w_train=False)
            val_loss = r_anode_mass_joint_untransformed(model_S=model_S,model_B=None,w=self.w_value,optimizer=optimizer,data_loader=valloader,
                                                        params=params, device=self.device, mode='val',
                                                        w_train=False)
            test_loss = r_anode_mass_joint_untransformed(model_S=model_S,model_B=None,w=self.w_value,optimizer=optimizer,data_loader=testloader,
                                                         params=params, device=self.device, mode='test',
                                                         w_train=False)

            torch.save(model_S.state_dict(), scrath_path+'/model_S/model_S_'+str(epoch)+'.pt')

            trainloss_list.append(train_loss)
            valloss_list.append(val_loss)
            testloss_list.append(test_loss)
            print('Epoch: ', epoch, 'Train loss: ', train_loss, 'Val loss: ', val_loss, 'Test loss: ', test_loss)

        # save train and val loss
        trainloss_list=np.array(trainloss_list)
        valloss_list=np.array(valloss_list)
        testloss_list=np.array(testloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)
        np.save(self.output()["testloss_list"].path, testloss_list)

        # save best models with lowest val loss
        best_models = np.argsort(valloss_list)[:self.num_model_to_save]
        for i in range(self.num_model_to_save):
            print(f'best model {i}: {best_models[i]}, valloss: {valloss_list[best_models[i]]}, testloss: {testloss_list[best_models[i]]}')
            os.rename(scrath_path+'/model_S/model_S_'+str(best_models[i])+'.pt', self.output()["sig_models"][i].path)

        # save metadata
        metadata = {"w_true": self.s_ratio, "num_train_events" : traintensor_S.shape[0], "num_val_events": valtensor_S.shape[0], "num_test_events": testtensor_S.shape[0]}
        metadata["min_val_loss_list"] = valloss_list[best_models]
        metadata["min_test_loss_list"] = testloss_list[best_models]
        metadata["min_train_loss_list"] = trainloss_list[best_models]

        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)


class ScanRANODEoverW(
    TemplateUncertaintyMixin,
    SignalStrengthMixin,
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

        val_loss = np.array([results[f"model_{index}"]["mean_valloss"] for index in range(self.scan_number)])
        val_loss_std = np.array([results[f"model_{index}"]["std_valloss"] for index in range(self.scan_number)])

        # save metadata
        w_best_index = np.argmin(val_loss)
        w_best = w_range[w_best_index]
        metadata = {"w_true": w_true, "w_best": w_best, "w_best_index": w_best_index}
        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)

        # TODO:
        # use test set not validation set?


class InterpolateRANODEoverW(
    ScanRANODEoverW
):
    """
    In previous task ScanRANODEoverW, we have scanned over w values and found the best w value, the likelihood plot
    is made by taking the mean of the val loss over all best models at each w value, and it will peak at w_best

    An alternative way of this will be take the best w, and the model_S at this w value, then simply plot
    w * model_S() + (1-w) * model_B() and compare with method 1

    The difference here is that in method 1, every point has its own model optimized for this w value, and the likelihood
    is the likelihood of the best model at this w value. In method 2, we take the best model at the best w value and simely
    change its w value in interpolation, so the model at different w values are always the same

    Not sure which one is correct
    """

    device = luigi.Parameter(default="cuda")

    def requires(self):
        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

        w_scan_results = ScanRANODEoverW.req(self)

        return {
            'preprocessing': Preprocessing.req(self),
            'bkgprob': PredictBkgProb.req(self),
            "models": model_list,
            "w_scan_results": w_scan_results,
        }

    def output(self):
        return {
            "comparison_plot": self.local_target("comparison_plot.pdf"),
            "metadata": self.local_target("metadata.json"),
        }

    @law.decorator.safe_output
    def run(self):
            
        # load previous scan result
        w_scan_results = self.input()["w_scan_results"]["metadata"].load()
        w_best = w_scan_results["w_best"]
        w_best_index = w_scan_results["w_best_index"]
        w_true = w_scan_results["w_true"]

        scan_results = self.input()["w_scan_results"]["scan_results"].load()
        w_scan_list = []
        likelihood_scan_list = []
        likelihood_scan_list_std = []
        for model_name, value in scan_results.items():
            w_scan_list.append(value["w"])
            likelihood_scan_list.append(-1 * value["mean_valloss"])
            likelihood_scan_list_std.append(value["std_valloss"])
            
        # load the model at best w
        model_best_list = []
        for rand_seed_index in range(len(self.input()["models"][f"model_{w_best_index}"])):
            model_best_seed_i = self.input()["models"][f"model_{w_best_index}"][rand_seed_index]["sig_models"]
            for model in model_best_seed_i:
                model_best_list.append(model.path)

        # define model
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from nflow_utils import flows_model_RQS
        

        # ------- load data -------

        # for model_B
        data_val_SR_B = np.load(self.input()['preprocessing']['data_val_SR_B'].path)
        data_val_SR_B_prob = np.load(self.input()['bkgprob']['log_B_val'].path) 
        log_B_val_tensor = torch.from_numpy(data_val_SR_B_prob.astype('float32')).to(self.device).flatten()

        # p(m) for model_B p(x|m)
        valtensor_B = torch.from_numpy(data_val_SR_B.astype('float32')).to(self.device)
        SR_mass_hist = np.load(self.input()['preprocessing']['SR_mass_hist'].path)
        SR_mass_bins = np.load(self.input()['preprocessing']['SR_mass_bins'].path)
        density_back = rv_histogram((SR_mass_hist, SR_mass_bins))
        val_mass_prob_B = torch.from_numpy(density_back.pdf(valtensor_B[:,0].cpu().detach().numpy())).to(self.device)

        bkg_likelihood = torch.exp(log_B_val_tensor)*val_mass_prob_B
        bkg_likelihood = bkg_likelihood.cpu().detach().numpy()

        # load and apply model_S
        signal_likelihood_list = []
        data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
        valtensor_S = torch.from_numpy(data_val_SR_S.astype('float32')).to(self.device)
        for model_S_dir in model_best_list:
            model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
            model_S.load_state_dict(torch.load(model_S_dir))
            model_S.eval()
            with torch.no_grad():
                log_S_val_tensor = model_S.log_prob(valtensor_S[:, :-1])
                S_likelihood = torch.exp(log_S_val_tensor)
                signal_likelihood_list.append(S_likelihood.cpu().detach().numpy())

            # clean cuda memory
            del model_S
            torch.cuda.empty_cache()

        signal_likelihood_list = np.array(signal_likelihood_list) # shape is (num_models, num_samples)
        
        w_scan_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), 1001)

        likelihood_interpolate = []
        likelihood_interpolate_std = []

        for w in w_scan_range:
            likelihood_interpolate_w = []
            for index, model_dir in enumerate(model_best_list):
                bkg_prob = np.array((1 - w) * bkg_likelihood)
                signal_prob = np.array(w * signal_likelihood_list[index])
                likelihood = np.log(signal_prob + bkg_prob + 1e-32)

                # ----------------- take mean of likelihood -----------------
                # in previous function r_anode_mass_joint_untransformed, the likelihood is calculated for each sample in a batch
                # have to do the same thing here to make numerically exact match between two methods on w_best model
                # reshape into batch and then take mean
                batchsize = 1024 * 5
                batch_num = likelihood.shape[0] // batchsize
                
                rest = likelihood[batch_num * batchsize:]
                likelihood = likelihood[:batch_num * batchsize].reshape(batch_num, batchsize)
                likelihood = np.mean(likelihood, axis=1)
                likelihood_rest = np.mean(rest)
                likelihood = np.append(likelihood, likelihood_rest)
                likelihood = np.mean(likelihood)
                likelihood_interpolate_w.append(likelihood)

            likelihood_interpolate_w = np.array(likelihood_interpolate_w)
            likelihood_interpolate.append(np.mean(likelihood_interpolate_w))
            likelihood_interpolate_std.append(np.std(likelihood_interpolate_w))


        likelihood_interpolate = np.array(likelihood_interpolate)
        likelihood_interpolate_std = np.array(likelihood_interpolate_std)

        # draw the uncertainty bin
        # we have 95% CI to be max likelihood - ln(2) / # val samples
        CI_95 = np.log(2) / signal_likelihood_list.shape[1]
        likelihood_interpolate_95 = max(likelihood_scan_list) - CI_95


        # save metadata
        metadata = {"w_best": w_best, "w_true": w_true, "likelihood_scan_list": likelihood_scan_list, 
                    "likelihood_scan_list_std": likelihood_scan_list_std, "likelihood_interpolate": likelihood_interpolate}
        with open(self.output()["metadata"].path, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)


        # plot
        self.output()["comparison_plot"].parent.touch()
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(self.output()["comparison_plot"].path) as pdf:

            # plot comparison
            f = plt.figure()
            plt.plot(np.log10(w_scan_list), likelihood_scan_list, color='r', label='w_scan result')
            plt.fill_between(np.log10(w_scan_list), np.array(likelihood_scan_list) - np.array(likelihood_scan_list_std),
                                np.array(likelihood_scan_list) + np.array(likelihood_scan_list_std), color='r', alpha=0.2)

            plt.plot(np.log10(w_scan_range), likelihood_interpolate, color='blue', label='interpolated result')
            plt.fill_between(np.log10(w_scan_range), likelihood_interpolate - likelihood_interpolate_std, 
                            likelihood_interpolate + likelihood_interpolate_std, color='blue', alpha=0.2)

            plt.axvline(np.log10(w_true), color='black', label='w_true')
            plt.scatter(np.log10(w_best), max(likelihood_scan_list), color='black', label='w_best')

            # draw 95 CI cut
            plt.plot(np.log10(w_scan_range), np.ones_like(w_scan_range) * likelihood_interpolate_95, color='black', linestyle='--', label='95% CI')

            plt.xlabel('log10(w)')
            plt.ylabel('likelihood')
            plt.title(f'w scan likelihood, w_best is {w_best:.5f}')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            # plot likelihood scan
            f = plt.figure()
            plt.plot(np.log10(w_scan_list), likelihood_scan_list, color='r', label='w_scan result')
            plt.fill_between(np.log10(w_scan_list), np.array(likelihood_scan_list) - np.array(likelihood_scan_list_std),
                                np.array(likelihood_scan_list) + np.array(likelihood_scan_list_std), color='r', alpha=0.2)
            
            plt.axvline(np.log10(w_true), color='black', label='w_true')
            plt.scatter(np.log10(w_best), max(likelihood_scan_list), color='black', label='w_best')

            # draw 95 CI cut
            plt.plot(np.log10(w_scan_range), np.ones_like(w_scan_range) * likelihood_interpolate_95, color='black', linestyle='--', label='95% CI')

            plt.xlabel('log10(w)')
            plt.ylabel('likelihood')
            plt.title(f'w scan likelihood, w_best is {w_best:.5f}')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            # plot interpolated likelihood
            f = plt.figure()
            plt.plot(np.log10(w_scan_range), likelihood_interpolate, color='blue', label='interpolated result')
            plt.fill_between(np.log10(w_scan_range), likelihood_interpolate - likelihood_interpolate_std,
                            likelihood_interpolate + likelihood_interpolate_std, color='blue', alpha=0.2)
            
            plt.axvline(np.log10(w_true), color='black', label='w_true')
            plt.scatter(np.log10(w_best), max(likelihood_scan_list), color='black', label='w_best')

            # draw 95 CI cut
            plt.plot(np.log10(w_scan_range), np.ones_like(w_scan_range) * likelihood_interpolate_95, color='black', linestyle='--', label='95% CI')

            plt.xlabel('log10(w)')
            plt.ylabel('likelihood')
            plt.title(f'interpolated likelihood, w_best is {w_best:.5f}')
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

        

        
class GenerateSignals(
   ScanRANODEoverW, 
):
    device = luigi.Parameter(default="cuda")
    n_signal_samples = luigi.IntParameter(default=10000)
    
    def requires(self):
        model_list = {}
        w_range = np.logspace(np.log10(self.w_min), np.log10(self.w_max), self.scan_number)

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = [RNodeTemplate.req(self, w_value=w_range[index], train_random_seed=i) for i in range(self.num_templates)]

        w_scan_results = ScanRANODEoverW.req(self)

        return {
            "models": model_list,
            "w_scan_results": w_scan_results,
        } 

    def output(self):
        return {
            "signal_list": self.local_target("signal_list.npy"),
        }

    @law.decorator.safe_output
    def run(self):

        # load previous scan result
        w_scan_results = self.input()["w_scan_results"]["metadata"].load()
        w_best = w_scan_results["w_best"]
        w_best_index = w_scan_results["w_best_index"]

        # load the model at best w
        model_best_list = []
        for rand_seed_index in range(len(self.input()["models"][f"model_{w_best_index}"])):
            model_best_seed_i = self.input()["models"][f"model_{w_best_index}"][rand_seed_index]["sig_models"]
            for model in model_best_seed_i:
                model_best_list.append(model.path)

        # define model
        ranode_path = os.environ.get("RANODE")
        sys.path.append(ranode_path)
        from nflow_utils import flows_model_RQS

        # generate signals from each models
        signal_list = []
        for model_dir in model_best_list:
            model_S = flows_model_RQS(device=self.device, num_features=5, context_features=None)
            model_S.load_state_dict(torch.load(model_dir))
            model_S.eval()

            signal_samples = model_S.sample(self.n_signal_samples)
            signal_list.append(signal_samples.cpu().detach().numpy())

            # clean cuda memory
            del model_S
            torch.cuda.empty_cache()

        # sample weight is 1 / num_models
        signal_list = np.array(signal_list)
        sample_weight = 1 / signal_list.shape[0]
        sample_weight = np.ones((len(signal_list), self.n_signal_samples, 1)) * sample_weight

        signal_list = np.concatenate([signal_list, sample_weight], axis=-1)

        signal_list = signal_list.reshape(-1, 6)

        self.output()["signal_list"].parent.touch()
        np.save(self.output()["signal_list"].path, signal_list)


class SignalGenerationPlot(
    GenerateSignals,
):
    nbins = luigi.IntParameter(default=41)

    def requires(self):
        return {
            "generated_signal_list": GenerateSignals.req(self, n_signal_samples=self.n_sig),
            "preprocessing": Preprocessing.req(self),
        }

    def output(self):
        return self.local_target("signal_plot.pdf")

    @law.decorator.safe_output
    def run(self):

        generated_signals = np.load(self.input()["generated_signal_list"]["signal_list"].path)
        generated_signal_features = generated_signals[:, 1:-1]
        generated_signal_weights = generated_signals[:, -1]
        generated_signals_mass = generated_signal_features[:, 0]

        # load data
        data_val_SR_S = np.load(self.input()['preprocessing']['data_val_SR_S'].path)
        mask_signals_val = data_val_SR_S[:, -1] == 1
        signal_val = data_val_SR_S[mask_signals_val]
        signal_val_features = signal_val[:, 1:-1]
        signal_val_mass = signal_val_features[:, 0]

        mask_bkg_val = data_val_SR_S[:, -1] == 0
        bkg_val = data_val_SR_S[mask_bkg_val]
        bkg_val_features = bkg_val[:, 1:-1]
        bkg_val_mass = bkg_val_features[:, 0]

        # plot
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        self.output().parent.touch()
        with PdfPages(self.output().path) as pdf:
            
            for feature_index in range(signal_val_features.shape[1]):

                bins = np.linspace(bkg_val_features[:, feature_index].min(), bkg_val_features[:, feature_index].max(), self.nbins)
                                   
                f = plt.figure()
                plt.hist(signal_val_features[:, feature_index], bins=bins, alpha=0.5, label='val signal', density=True, histtype='step', lw=3)
                plt.hist(bkg_val_features[:, feature_index], bins=bins, alpha=0.5, label='val bkg', density=True, histtype='step', lw=3)
                plt.hist(generated_signal_features[:, feature_index], bins=bins, weights=generated_signal_weights, 
                         alpha=0.5, label='generated signal', density=True, histtype='step', lw=3)
                plt.xlabel(f'feature {feature_index}')
                plt.ylabel('density')
                plt.title(f'feature {feature_index} distribution, {self.n_sig} signal in samples')
                plt.legend()
                plt.yscale('log')
                pdf.savefig(f)
                plt.close(f)



