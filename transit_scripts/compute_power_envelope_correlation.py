import os.path as op
from utils import make_envelope_correllation
import mne

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

duration = 30.
stop = None
overlap = 8.

raw_label = mne.io.read_raw_fif('labels_aparc_sk_broadband_raw.fif', preload=True)

env_corr = make_envelope_correllation(
    raw_label=raw_label, duration=duration, overlap=overlap, stop=stop, fmin=13, fmax=30)

import matplotlib.pyplot as plt
import numpy as np
plt.matshow(np.abs(env_corr.mean(0)), cmap='viridis')
plt.colorbar()

mne.externals.h5io.write_hdf5('beta_band_brain_power_envelopes_beta_wide.h5', {'band': 'beta', 'C': env_corr})
