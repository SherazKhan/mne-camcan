import copy
import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne
import numpy as np

mne.utils.set_log_level('warning')


def decimate_raw(raw, decim):
    new_lowpass = raw.info['sfreq'] / decim
    raw._data = raw.get_data()[:, ::decim]
    raw._times = raw._times[::decim]
    raw.info['sfreq'] = new_lowpass
    raw._last_samps[0] /= decim
    raw._first_samps[0] /= decim


def make_surrogates_empty_room(raw, fwd, inverse_operator, step=10000):
    """Create spatially structured noise from empty room MEG

    .. note::
        Convert MEG empty room to spatially structured noise by applying
        the inverse solution and then the forward solution.

    .. note::
        To safe memory, projection proceeds in non-overlapping sliding windows.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data (empty room).
    fwd : instance of mne.Forward
        The forward solution
    inverse_operator : mne.minimum_norm.InverseOperator
        The inverse solution.
    step : int
        The step size (in samples) when iterating over the raw object.

    Returns
    -------
    raw_surr : instance of mne.io.Raw
        The surrogate MEG data.
    """
    index = np.arange(len(raw.times)).astype(int)
    out = np.empty(raw.get_data().shape, dtype=raw.get_data().dtype)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
    other_picks = [ii for ii in range(len(raw.ch_names)) if ii not in picks]
    out[other_picks] = raw.get_data()[other_picks]
    last = len(index)
    for start in index[::step]:
        stop = start + min(step, last - start)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        reprojected = mne.apply_forward(fwd=fwd, stc=stc, info=raw.info)
        out[picks, start:stop] = reprojected.data
    out = mne.io.RawArray(out, info=copy.deepcopy(raw.info))
    return out


def get_label_envelopes(raw, labels, inverse_operator, step=10000,
                        source_decim=10):
    """Create power envelopes for set of labels

    .. note::
        To safe memory, projection proceeds in non-overlapping sliding windows.

    .. note::
        This can take some time (scales linearly with number of samples,
        time points and dipoles).

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data (empty room).
    labels : list of mne.Label objects
        The labels to be visited
    inverse_operator : mne.minimum_norm.InverseOperator
        The inverse solution.
    step : int (defaults to 10000)
        The step size in sample when iterating over the raw object.
    source_decim : int
        The decimation factor on output data

    Returns
    -------
    X_surr : np.ndarray of shape(n_labels, n_times / source_decim)
        The surrogate MEG data.
    """
    n_source_times = raw.get_data().shape[1] / source_decim
    X_stc = np.empty((len(labels), n_source_times), dtype=np.float)
    index = np.arange(len(raw.times)).astype(int)
    sfreq = raw.info['sfreq'] / source_decim
    src = inverse_operator['src']
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
            start_target = int(start // source_decim)
            stop_target = int(stop // source_decim)
            X_stc[label_idx][start_target:stop_target] = tc
    return X_stc, sfreq


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
raw.apply_hilbert(envelope=False)
labels = [ll for ll in labels if 'unknown' not in ll.name]

X_stc, sfreq_env_data = get_label_envelopes(
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
raw_repro = make_surrogates_empty_room(raw_noise, fwd_fixed, inverse_operator)

raw_repro.filter(
   14, 30, l_trans_bandwidth=1., h_trans_bandwidth=1.,
   filter_length='auto', phase='zero', fir_window='hann')
raw_repro.apply_hilbert(envelope=False)

X_stc_noise, sfreq_env_noise = get_label_envelopes(
    raw_repro, labels, inverse_operator, step=10000)

mne.filter.filter_data(X_stc_noise, sfreq_env_noise, 0, 1)

mne.externals.h5io.write_hdf5(
    'power_envelopes_noise.h5',
    {'beta': X_stc_noise, 'sfreq': sfreq_env_noise})
