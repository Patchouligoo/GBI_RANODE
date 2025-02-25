#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle
from config.configs import SR_MAX, SR_MIN


def separate_SB_SR(data):
    innermask = (data[:, 0] > SR_MIN) & (data[:, 0] < SR_MAX)
    outermask = ~innermask
    return data[innermask], data[outermask]


def resample_split_trainval(signal_path, bkg_path, sig_ratio=0.005, resample_seed = 42):

    background = np.load(bkg_path)
    signal = np.load(signal_path)

    # shuffle data
    background = shuffle(background, random_state=resample_seed)
    signal = shuffle(signal, random_state=resample_seed)

    # split bkg into SR and CR
    SR_bkg, CR_bkg = separate_SB_SR(background)

    SR_sig, CR_sig = separate_SB_SR(signal)
    # for now we ignore signal in CR
    
    # calculate the amount of signal we inject
    num_sig = int(sig_ratio/(1-sig_ratio) * len(SR_bkg))
    
    SR_sig_injected = SR_sig[:num_sig]

    # concatenate background and signal
    SR_data_trainval = np.concatenate((SR_bkg, SR_sig_injected),axis=0)
    SR_data_trainval = shuffle(SR_data_trainval, random_state=resample_seed)
    SR_data_train, SR_data_val = train_test_split(SR_data_trainval, test_size=0.2, random_state=resample_seed)

    true_mu_train = (SR_data_train[:, -1]==1).sum() / len(SR_data_train)
    true_mu_val = (SR_data_val[:, -1]==1).sum() / len(SR_data_val)

    print('SR train shape: ', SR_data_train.shape)
    print('SR train num sig: ', (SR_data_train[:, -1]==1).sum())
    print('SR train true mu: ', true_mu_train)

    print('SR val shape: ', SR_data_val.shape)
    print('SR val true mu: ', true_mu_val)
    print('SR val num sig: ', (SR_data_val[:, -1]==1).sum())

    print('CR shape: ', CR_bkg.shape)

    return SR_data_train, SR_data_val, CR_bkg


def resample_split_test(signal_path, bkg_path, resample_seed = 42):

    background = np.load(bkg_path)
    signal = np.load(signal_path)

    # shuffle data
    background = shuffle(background, random_state=resample_seed)
    signal = shuffle(signal, random_state=resample_seed)

    # split bkg into SR and CR
    SR_bkg, CR_bkg = separate_SB_SR(background)

    SR_sig, CR_sig = separate_SB_SR(signal)
    # for now we ignore signal in CR
    
    SR_sig_injected = SR_sig[:50000]

    # concatenate background and signal
    SR_data_test = np.concatenate((SR_bkg, SR_sig_injected),axis=0)
    SR_data_test = shuffle(SR_data_test, random_state=resample_seed)

    print('SR test shape: ', SR_data_test.shape)
    print('SR test num sig: ', (SR_data_test[:, -1]==1).sum())

    return SR_data_test

