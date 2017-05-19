# cd transit_scripts/
import glob
import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne

from utils import (
    decimate_raw, make_surrogates_empty_room, compute_inverse_raw_label)

mne.utils.set_log_level('warning')

data_path = bst_resting.data_path()

raw_noise_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_noise_20111104_02_comp_raw.fif')

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

spacing = 'ico5'

noise_cov_fname = op.join(
    meg_dir, subject, '%s-%s-cov.fif' % (subject, spacing))
fwd_fname = op.join(meg_dir, subject, '%s_%s-fwd.fif' % (subject, spacing))

raw = mne.io.read_raw_fif('brainstorm_testing_rest_raw.fif')
raw.load_data()
raw.pick_types(meg=True, eeg=False)
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']
noise_cov = mne.read_cov(noise_cov_fname)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov)

labels = [mne.read_label(fpath) for fpath in glob.glob('./testing_labels/*label')]

raw_label = compute_inverse_raw_label(raw, labels, inverse_operator,
                                      label_mode='pca_flip_truncated')
raw_label.rename_channels(
    dict(zip(raw_label.ch_names,
             ['l%03d' % ii for ii, _ in enumerate(raw_label.ch_names)])))

raw_label.save('labels_aparc_sk_broadband_raw.fif', overwrite=True)

del raw_label

raw_noise = mne.io.read_raw_fif(raw_noise_fname)
projs = mne.read_proj(proj_fname)
raw_noise.rename_channels(
    dict(zip(raw_noise.ch_names,
             mne.utils._clean_names(raw_noise.ch_names))))

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
decimate_raw(raw_noise, decim=24)
raw_repro = make_surrogates_empty_room(raw_noise, fwd_fixed, inverse_operator)

raw_label_noise = compute_inverse_raw_label(
    raw_repro, labels, inverse_operator, label_mode='pca_flip_mean')

raw_label_noise.save('labels_aparc_sk_broadband_noise_raw.fif', overwrite=True)
