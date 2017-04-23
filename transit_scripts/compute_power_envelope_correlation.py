import os.path as op
from utils import make_envelope_correllation
import mne

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

duration = 30.
stop = 100.
overlap = 30.

for fname in ('power_envelopes.h5', 'power_envelopes_noise.h5'):
    h5data = mne.externals.h5io.read_hdf5(fname)
    stcs = h5data['beta']
    sfreq = h5data['sfreq']
    env_corr = make_envelope_correllation(
        stcs=stcs, duration=duration, overlap=overlap, stop=stop, sfreq=sfreq)

    mne.externals.h5io.write_hdf5(
        fname.replace('.h5', '_corr.h5'), env_corr, overwrite=True)
    break

env_corr[0].flat[::len(env_corr[0]) + 1] = 0.

import matplotlib.pyplot as plt
import numpy as np


plt.matshow(np.abs(env_corr[0]), cmap='viridis')
