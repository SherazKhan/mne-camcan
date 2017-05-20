import os.path as op
from utils import make_envelope_correllation
import mne

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

duration = 30.
stop = 100.
overlap = 30.

raw_label = mne.io.read_raw_fif('labels_aparc_sk_broadband_raw.fif', preload=True)

env_corr = make_envelope_correllation(
    raw_label=raw_label, duration=duration, overlap=overlap, stop=stop, fmin=13, fmax=20)

# env_corr[0].flat[::len(env_corr[0]) + 1] = 0.
#
import matplotlib.pyplot as plt
import numpy as np
plt.matshow(np.abs(env_corr[0]), cmap='viridis')
plt.colorbar()
