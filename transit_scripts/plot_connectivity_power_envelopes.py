import os.path as op
from utils import compute_envelope_correllation, make_overlapping_events

import mne
import numpy as np
from scipy.signal import hilbert

h5data = mne.externals.h5io.read_hdf5('power_envelopes.h5')
X_beta_env = h5data['beta']
sfreq = h5data['sfreq']

X_beta_env_ = hilbert(X_beta_env, axis=-1)

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

labels = mne.read_labels_from_annot(
    parc='aparc_sk', subject='fsaverage', subjects_dir=subjects_dir)

ch_names = [l.name for l in labels if 'unknown' not in l.name]

info = mne.create_info(
    ch_names, sfreq=sfreq, ch_types=['misc'] * len(X_beta_env_))

stc_raw = mne.io.RawArray(X_beta_env_, info)

events = make_overlapping_events(stc_raw, 3000, duration=30, overlap=5,
                                 stop=300.)

stc_epochs = mne.Epochs(stc_raw, events=events, tmin=0, tmax=30, baseline=None,
                        reject=None, preload=True)

orth_corrs = np.empty((len(stc_epochs), len(ch_names), len(ch_names)),
                      dtype=np.float)

for ii, epoch in enumerate(stc_epochs):
    orth_corrs[ii] = compute_envelope_correllation(epoch)
