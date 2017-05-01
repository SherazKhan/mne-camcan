import os.path as op
from utils import compute_envelope_correllation, make_overlapping_events

import mne
import numpy as np

raw_label = mne.io.read_raw_fif('labels_aparc_sk_broadband_raw.fif',
                                preload=True)

picks = mne.pick_types(raw_label.info, misc=True, meg=False)

raw_label.filter(14, 30, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                 picks=picks,
                 filter_length='auto', phase='zero', fir_window='hann')
raw_label.apply_hilbert(picks=picks, envelope=False)

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

events = make_overlapping_events(raw_label, 3000, duration=30, overlap=5,
                                 stop=300.)

stc_epochs = mne.Epochs(raw_label, events=events, tmin=0, tmax=30,
                        baseline=None, reject=None, preload=True)

n_chans = raw_label.info['nchan']
orth_corrs = np.empty((len(stc_epochs), n_chans, n_chans),
                      dtype=np.float)
X = stc_epochs.get_data()

for ii, epoch in enumerate(stc_epochs):
    corr = compute_envelope_correllation(epoch)
    orth_corrs[ii] = corr

mne.externals.h5io.write_hdf5('beta_band_connectivity.h5', orth_corrs,
                              overwrite=True)
