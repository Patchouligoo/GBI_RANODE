#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle
from config.configs import SR_MAX, SR_MIN


def separate_SB_SR(data):
    innermask = (data[:, 0] > SR_MIN) & (data[:, 0] < SR_MAX)
    outermask = ~innermask
    return data[innermask], data[outermask]


def resample_split(signal_path, bkg_path, sig_ratio=0.005 , bkg_num_in_sr_data=-1, resample_seed = 42):

    background = np.load(bkg_path)
    signal = np.load(signal_path)

    # shuffle data
    background = shuffle(background, random_state=resample_seed)
    signal = shuffle(signal, random_state=resample_seed)

    # split bkg into SR and CR
    SR_bkg, CR_bkg = separate_SB_SR(background)

    SR_sig, CR_sig = separate_SB_SR(signal)
    # for now we ignore signal in CR
    
    # have to make the same number of events as PAWS does for fair comparison
    if bkg_num_in_sr_data != -1:
        # this includes 50% for training and 25%, 25% for val and test
        SR_bkg = SR_bkg[:bkg_num_in_sr_data]

    # split into train, val, test in 50:25:25
    SR_bkg_train, SR_bkg_val = train_test_split(SR_bkg, test_size=0.5, random_state=resample_seed)
    SR_bkg_val, SR_bkg_test = train_test_split(SR_bkg_val, test_size=0.5, random_state=resample_seed)
    len_train_val = len(SR_bkg_train) + len(SR_bkg_val)

    # calculate the amount of signal we inject
    num_sig = int(sig_ratio/(1-sig_ratio) * len_train_val)
    
    SR_sig_injected = SR_sig[:num_sig]
    SR_sig_testset = SR_sig[num_sig:][:50000] # take 50k for testing

    # concatenate background and signal
    SR_data_trainval = np.concatenate((SR_bkg_train, SR_bkg_val, SR_sig_injected),axis=0)
    SR_data_trainval = shuffle(SR_data_trainval, random_state=resample_seed)
    SR_data_train, SR_data_val = train_test_split(SR_data_trainval, test_size=1.0/3, random_state=resample_seed)

    SR_data_test = np.concatenate((SR_bkg_test, SR_sig_testset),axis=0)

    true_mu_train = (SR_data_train[:, -1]==1).sum() / len(SR_data_train)
    true_mu_val = (SR_data_val[:, -1]==1).sum() / len(SR_data_val)

    print('SR train shape: ', SR_data_train.shape)
    print('SR train num sig: ', (SR_data_train[:, -1]==1).sum())
    print('SR train true mu: ', true_mu_train)

    print('SR val shape: ', SR_data_val.shape)
    print('SR val true mu: ', true_mu_val)
    print('SR val num sig: ', (SR_data_val[:, -1]==1).sum())

    print('SR test shape: ', SR_data_test.shape)
    print('SR test num sig: ', (SR_data_test[:, -1]==1).sum())

    print('CR shape: ', CR_bkg.shape)

    return SR_data_train, SR_data_val, SR_data_test, CR_bkg

