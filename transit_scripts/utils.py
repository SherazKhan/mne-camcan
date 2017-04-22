import copy
import mne
import numpy as np


def make_overlapping_events(raw, event_id, overlap, duration,
                            stop=None):
    """Create overlapping events"""
    if stop is None:
        stop = raw.times[raw.last_samp]
    events = list()
    for start in np.arange(0, duration, overlap):
        events.append(mne.make_fixed_length_events(
            raw, id=event_id, start=start, stop=stop, duration=duration))
    events_max = events[0][:, 0].max()
    events = [e[np.where(e[:, 0] <= events_max)] for e in events]
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]

    return events


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


def get_label_data(raw, labels, inverse_operator, step=10000,
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
            tc = tc[0, ::source_decim]
            start_target = int(start // source_decim)
            stop_target = int(stop // source_decim)
            X_stc[label_idx][start_target:stop_target] = tc
    return X_stc, sfreq


def compute_corr(x, y):
    """Correlate 2 matrices along last axis"""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r


def orthogonalize_y(x, y):
    pow_x = np.abs(x)
    pow_x **= 2
    y_ = y - np.real((x * y.conjugate()) / (pow_x))
    y_ *= x
    return y_
    # set y to x and make sure y_oth is 0


def orthogonalize_x(x, y):
    return orthogonalize_y(y, x)


def filt_fun(x, sfreq):
    return mne.filter.filter_data(
        x, sfreq=sfreq, l_freq=0, h_freq=1, h_trans_bandwidth=0.1,
        fir_design='firwin2')


def compute_envelope_correllation(X):
    """"""
    output = np.empty((len(X), len(X)), dtype=np.float)
    y = X

    def fun(x):
        return np.abs(x)

    for ii, x in enumerate(X):
        x_ = orthogonalize_x(x, y)
        y_ = orthogonalize_y(x, y)
        output[ii] = np.mean((
            compute_corr(fun(x), fun(y_)),
            compute_corr(fun(x_), fun(y))), 0)
    return output
