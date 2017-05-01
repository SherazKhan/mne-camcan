import copy
import mne
from mne.io.constants import FIFF
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


def _get_label_data(raw, labels, inverse_operator, step=10000,
                    source_decim=10, label_mode='pca_flip'):
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
    label_mode : str (defaults to 'pca_flip')
        The method to extract one time course from a label.

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
    # XXX return Raw here actually and call it `compute_inverse_raw`
    for start in index[::step]:
        stop = start + min(step, last - start)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        for label_idx, label in enumerate(labels):
            tc = stc.extract_label_time_course(
                label, src, mode=label_mode)
            tc = tc[0, ::source_decim]
            start_target = int(start // source_decim)
            stop_target = int(stop // source_decim)
            X_stc[label_idx][start_target:stop_target] = tc
    return X_stc, sfreq


def compute_inverse_raw_label(raw, labels, inverse_operator, step=10000,
                              label_mode='pca_flip'):
    """Compute inverse solution label time series as channels in raw object

    .. note::
        To safe memory, projection proceeds in non-overlapping sliding windows.

    .. note::
        This can take some time (scales linearly with number of samples,
        time points and dipoles).

    .. note::
        Label names are too long to be saved into the 32bit FIF format.
        Get your label names from the list of labels that you passed here.
        Here, channel will be named L001, L002, etc.

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
    label_mode : str (defaults to 'pca_flip')
        The method to extract one time course from a label.

    Returns
    -------
    label_raw : instance of mne.RawArray
        A raw object with continous data where channels are the continous data
        summarized for a given label.
    """
    label_raw, sfreq = _get_label_data(
        raw=raw, labels=labels, inverse_operator=inverse_operator, step=step,
        label_mode=label_mode, source_decim=1)

    n_labels = len(labels)
    label_names = ['L%03d' % ii for ii in range(n_labels)]

    info = mne.create_info(
        ch_names=label_names, sfreq=sfreq, ch_types=['misc'] * n_labels)

    info['highpass'] = raw.info['highpass']
    info['lowpass'] = raw.info['lowpass']
    for ch in info['chs']:
        ch['unit'] = FIFF.FIFF_UNIT_AM  # put ampere meter
        ch['unit_mul'] = FIFF.FIFF_UNIT_NONE  # no unit multiplication

    label_raw = mne.io.RawArray(label_raw, info)
    label_raw.annotations = raw.annotations
    return label_raw


def compute_corr(x, y):
    """Correlate 2 matrices along last axis.

    Parameters
    ----------
    x : np.ndarray of shape(n_time_series, n_times)
        The first set of vectors.
    y : np.ndarray of shape(n_time_series, n_times)
        The second set of vectors.

    Retrurns
    --------
    r : np.ndarray of shape(n_time_series,)
        The correlation betwen x and y.
    """
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r


def _orthogonalize(a, b):
    """orthogonalize x on y."""
    return np.imag(a * (b.conj() / np.abs(b)))


def compute_envelope_correllation(X):
    """Compute power envelope correlation with orthogonalization.

    Parameters
    ----------
    X : np.ndarray of shape(n_labels, n_time_series)
        The source data.

    Returns
    -------
    corr : np.ndarray of shape (n_labels, n_labels)
        The connectivity matrix.
    """
    n_features = X.shape[0]
    corr = np.zeros((n_features, n_features), dtype=np.float)
    for ii, x in enumerate(X):
        jj = ii + 1
        y = X[jj:]
        x_, y_ = _orthogonalize(a=x, b=y), _orthogonalize(a=y, b=x)
        this_corr = np.mean((
            np.abs(compute_corr(np.abs(x), y_)),
            np.abs(compute_corr(np.abs(y), x_))), axis=0)
        corr[ii:jj, jj:] = this_corr

    corr.flat[::n_features + 1] = 0  # orthogonalized correlation should be 0
    return corr + corr.T  # mirror lower diagonal


def make_envelope_correllation(stcs, duration, overlap, stop, sfreq):

    label_names = [str(k) for k in range(len(stcs))]
    n_labels = len(label_names)

    info = mne.create_info(
        ch_names=label_names, sfreq=sfreq, ch_types=['misc'] * len(stcs))
    for ch in info['chs']:
        ch['unit'] = FIFF.FIFF_UNIT_AM
        ch['unit_mul'] = FIFF.FIFF_UNIT_NONE

    stcs = mne.io.RawArray(stcs, info)
    stcs.apply_hilbert(envelope=False, picks=list(range(n_labels)))

    events = make_overlapping_events(stcs, 3000, duration=duration,
                                     overlap=overlap, stop=stop)

    stcs = mne.Epochs(stcs, events=events, tmin=0, tmax=duration,
                      baseline=None, reject=None, preload=True)

    env_corrs = np.empty((len(stcs), n_labels, n_labels),
                         dtype=np.float)

    for ii, stc_epoch in enumerate(stcs):
        env_corrs[ii] = compute_envelope_correllation(stc_epoch)

    return env_corrs
