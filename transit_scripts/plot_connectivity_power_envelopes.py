import os.path as op
import glob
import math

import numpy as np

import mne
from mne.datasets.brainstorm import bst_resting

data_path = bst_resting.data_path()
raw_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_spontaneous_20111102_02_AUX_raw.fif')

raw_noise_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_noise_20111104_02_raw.fif')

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

spacing = 'ico5'

raw = mne.io.read_raw_fif(raw_fname)
raw.load_data()
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
raw.pick_types(meg=True, eog=True, ecg=True)

# ssp_ecg = mne.preprocessing.compute_proj_ecg(raw, n_jobs=4)
raw.filter(1, 200)
raw.info['highpass']
average_ecg = mne.preprocessing.create_ecg_epochs(raw).average()
average_ecg.apply_baseline(baseline=(-0.5, -0.3))

average_eog = mne.preprocessing.create_eog_epochs(raw).average()
average_eog.apply_baseline(baseline=(-0.5, -0.3))


# fast artefact rejection: project out the first EOG/ECG vectors
ssp_ecg, ecg_events = mne.preprocessing.compute_proj_ecg(raw)
ssp_eog, eog_events = mne.preprocessing.compute_proj_eog(raw)
raw.add_proj(ssp_ecg[0])
raw.add_proj(ssp_eog[0])

bem_fname = op.join(subjects_dir, subject, 'bem', '%s-bem.fif' % subject)
src_fname = op.join(
    meg_dir, subject, '%s-%s-src.fif' % (subject, spacing))

src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)

trans_fname = op.join(meg_dir, subject, "bs_resting_01-trans.fif")

fwd = mne.make_forward_solution(raw.info, trans_fname, src=src, bem=bem)

raw_noise = mne.io.read_raw_fif(raw_noise_fname)
raw_noise.add_proj(ssp_ecg[:1] + ssp_eog[:1])
raw_noise.load_data()
raw_noise.filter(1, 200)
noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')

whos
