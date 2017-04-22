import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne

from utils import decimate_raw, make_surrogates_empty_room, get_label_data

mne.utils.set_log_level('warning')

data_path = bst_resting.data_path()
raw_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_spontaneous_20111102_02_AUX_raw.fif')

raw_noise_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_noise_20111104_02_comp_raw.fif')

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

spacing = 'ico5'

proj_fname = op.join(meg_dir, subject, 'bst_resting_ecg-eog-proj.fif')
noise_cov_fname = op.join(
    meg_dir, subject, '%s-%s-cov.fif' % (subject, spacing))
fwd_fname = op.join(meg_dir, subject, '%s_%s-fwd.fif' % (subject, spacing))

raw = mne.io.read_raw_fif(raw_fname)
projs = mne.read_proj(proj_fname)
raw.add_proj(projs)
raw.rename_channels(
    dict(zip(raw.ch_names, mne.utils._clean_names(raw.ch_names))))
raw.load_data()
raw.pick_types(meg=True, eeg=False, ref_meg=False)

fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']
noise_cov = mne.read_cov(noise_cov_fname)

for comp in raw.info['comps']:
    for key in ('row_names', 'col_names'):
        comp['data'][key] = mne.utils._clean_names(comp['data'][key])

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov)

labels = mne.read_labels_from_annot(
    parc='aparc_sk', subject='fsaverage', subjects_dir=subjects_dir)

for label in labels:
    label.morph(subject_to=subject, subjects_dir=subjects_dir)


decimate_raw(raw, decim=9)
raw.filter(14, 30, l_trans_bandwidth=1., h_trans_bandwidth=1.,
           filter_length='auto', phase='zero', fir_window='hann')
labels = [ll for ll in labels if 'unknown' not in ll.name]

X_stc, sfreq_env_data = get_label_data(
    raw, labels, inverse_operator, step=10000)
mne.externals.h5io.write_hdf5(
    'power_envelopes.h5', {'beta': X_stc, 'sfreq': sfreq_env_data},
    overwrite=True)

del X_stc
del raw

raw_noise = mne.io.read_raw_fif(raw_noise_fname)
raw_noise.rename_channels(
    dict(zip(raw_noise.ch_names,
             mne.utils._clean_names(raw_noise.ch_names))))

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
decimate_raw(raw_noise, decim=24)
raw_repro = make_surrogates_empty_room(
    raw_noise, fwd_fixed, inverse_operator)

raw_repro.filter(
   14, 30, l_trans_bandwidth=1., h_trans_bandwidth=1.,
   filter_length='auto', phase='zero', fir_window='hann')

X_stc_noise, sfreq_env_noise = get_label_data(
    raw_repro, labels, inverse_operator, step=10000)

mne.externals.h5io.write_hdf5(
    'power_envelopes_noise.h5',
    {'beta': X_stc_noise, 'sfreq': sfreq_env_noise},
    overwrite=True)
