# Copyright (C) Denis A. Engemann - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Denis A. Engemann <denis.engemann@gmail.com>, May 2017
#
# Depends on and uses MNE-Python code (BSD-3-clause)

import numpy as np
import mne
from mne.constants import FIFF


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
