import os.path as op

import mne
from mne.datasets.brainstorm import bst_resting

data_path = bst_resting.data_path()
raw_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_spontaneous_20111102_02_AUX_raw.fif')

subject = 'bst_resting'
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

raw = mne.io.read_raw_fif(raw_fname)
raw.load_data()
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
raw.pick_types(meg=True, eog=True, ecg=True)

# ssp_ecg = mne.preprocessing.compute_proj_ecg(raw, n_jobs=4)
raw.filter(1, 200)

# fast artefact rejection: project out the first EOG/ECG vectors
ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(raw)
ssp_eog, _ = mne.preprocessing.compute_proj_eog(raw)

mne.write_proj(op.join(meg_dir, subject, '%s_ecg-eog-proj.fif' % subject),
               ssp_ecg[:1] + ssp_eog[:1])
