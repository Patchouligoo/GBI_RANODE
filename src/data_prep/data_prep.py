#import pandas as pd
import numpy as np
import os
import argparse
#import vector
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle


def separate_SB_SR(data, minmass, maxmass):
    innermask = (data[:, 0] > minmass) & (data[:, 0] < maxmass)
    outermask = ~innermask
    return data[innermask], data[outermask]


def resample_split(data_dir, sig_ratio=0.005 , resample_seed = 42,\
                   minmass = 3.3, maxmass = 3.7):
    background = np.load(f'{data_dir}/data_bg.npy')
    signal = np.load(f'{data_dir}/data_sig.npy')

    # split bkg into SR and CR
    SR_data, CR_data = separate_SB_SR(background, minmass, maxmass)

    num_bkg_SR = len(SR_data)
    num_bkg_CR = len(CR_data)

    # compute num signal events to match signal ratio
    num_sig = int(sig_ratio/(1-sig_ratio) * num_bkg_SR)
    # randomly choose num_sig signal events
    np.random.seed(resample_seed)
    choice = np.random.choice(len(signal), num_sig, replace=False)
    signal = signal[choice]

    # concatenate background and signal
    SR_data = np.concatenate((SR_data, signal),axis=0)
    SR_data = shuffle(SR_data, random_state=resample_seed)

    CR_data = shuffle(CR_data, random_state=resample_seed)
    
    S = SR_data[SR_data[:, -1]==1]
    B = SR_data[SR_data[:, -1]==0]

    true_w = len(S)/(len(B)+len(S))
    
    print('SR shape: ', SR_data.shape)
    print("num sig in SR: ", len(S))
    print("num bkg in SR: ", len(B))
    print('CR shape: ', CR_data.shape)

    return SR_data, CR_data, true_w
