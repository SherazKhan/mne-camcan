import os.path as op
import mne
import matplotlib.pyplot as plt
import numpy as np


subjects = ['090421', '090430', '090707']

data_path = '/home/sheraz/Dropbox/mne-camcan-data'
subject = subjects[0]

def compute_noise_cov(subject):
    raw_empty_room_fname = op.join(
        data_path, 'emptyroom', subject + '_raw_st.fif')
    raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname)

    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
    return noise_cov.data

data = np.array([compute_noise_cov(subject) for subject in subjects])

