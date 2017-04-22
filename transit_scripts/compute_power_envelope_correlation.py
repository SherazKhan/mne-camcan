import os.path as op
from utils import compute_envelope_correllation, make_overlapping_events

import mne
import numpy as np

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')


def make_envelope_correllation(stcs, duration, overlap, stop, sfreq):

    label_names = [str(k) for k in range(len(stcs))]
    n_labels = len(label_names)
    info = mne.create_info(
        ch_names=label_names, sfreq=sfreq, ch_types=['misc'] * len(stcs))

    stcs = mne.io.RawArray(stcs, info)
    stcs.apply_hilbert(envelope=False)

    events = make_overlapping_events(stcs, 3000, duration=duration,
                                     overlap=overlap, stop=stop)

    stcs = mne.Epochs(stcs, events=events, tmin=0, tmax=30,
                      baseline=None, reject=None, preload=True)

    env_corrs = np.empty((len(stcs), n_labels, n_labels),
                         dtype=np.float)

    for ii, stc_epoch in enumerate(stcs):
        env_corrs[ii] = compute_envelope_correllation(stc_epoch)

    return env_corrs


for fname in ('power_envelopes.h5', 'power_envelopes_noise.h5'):
    h5data = mne.externals.h5io.read_hdf5(fname)
    X_beta_env = h5data['beta']
    sfreq = h5data['sfreq']
    env_corr = make_envelope_correllation(
        stcs=X_beta_env, duration=30., overlap=5., stop=300., sfreq=sfreq)
    mne.externals.h5io.write_hdf5(fname.replace('.h5', '_corr.h5'), env_corr)
