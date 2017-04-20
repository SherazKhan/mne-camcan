import math
import copy
import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne
import numpy as np

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

decim = 9
new_lowpass = raw.info['sfreq'] / decim
raw._data = raw.get_data()[:, ::decim]
raw._times = raw._times[::decim]
raw.info['sfreq'] = new_lowpass
raw._last_samps[0] /= decim
raw._first_samps[0] /= decim
raw.filter(14, 30, l_trans_bandwidth=1., h_trans_bandwidth=1.,
           filter_length='auto', phase='zero', fir_window='hann')
raw.apply_hilbert(envelope=False)
labels = [ll for ll in labels if 'unknown' not in ll.name]


def reproject_raw(raw, fwd, inverse_operator, step=10000):
    index = np.arange(len(raw.times)).astype(int)
    last = len(index)
    out = np.empty(raw.get_data().shape, dtype=raw.get_data().dtype)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
    other_picks = [ii for ii in range(len(raw.ch_names)) if ii not in picks]
    out[other_picks] = raw.get_data()[other_picks]
    for start in index[::step]:
        stop = start + min(step, last - start)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        reprojected = mne.apply_forward(fwd=fwd, stc=stc, info=raw.info)
        out[picks, start:stop] = reprojected.data
    out = mne.io.RawArray(out, info=copy.deepcopy(raw.info))
    return out


raw_noise = mne.io.read_raw_fif(raw_noise_fname)
raw_noise.rename_channels(
    dict(zip(raw_noise.ch_names,
             mne.utils._clean_names(raw_noise.ch_names))))

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
raw_repro = reproject_raw(raw_noise, fwd_fixed, inverse_operator)


def get_label_envelopes(raw, labels, inverse_operator, step=10000,
                        source_decim=10):
    n_source_times = raw.get_data().shape[1] / source_decim
    X_stc = np.empty((len(labels), n_source_times), dtype=np.float)
    index = np.arange(len(raw.times)).astype(int)
    last = len(index)
    for start in index[::step]:
        stop = start + min(step, last - start)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        for label_idx, label in enumerate(labels):
            tc = stc.extract_label_time_course(
                label, src, mode='pca_flip_mean')
            tc = np.abs(tc[0, ::source_decim])
            start_target = int(math.floor(start / source_decim))
            stop_target = int(math.floor(stop / source_decim))
            X_stc[label_idx][start_target:stop_target] = tc
    return X_stc


X_stc = get_label_envelopes(raw, labels, step=10000)
mne.externals.h5io.write_hdf5('power_envelopes_beta.h5', {'X_stc': X_stc})
