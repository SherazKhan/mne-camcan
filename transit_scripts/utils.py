import copy
import mne
from mne.io.constants import FIFF
import numpy as np
from scipy import linalg
from mne.utils import warn as warn_, logger
from mne import SourceEstimate


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


def _get_label_flip(labels, label_vertidx, src):
    """Get sign-flip for labels."""
    # do the import here to avoid circular dependency
    from mne.label import label_sign_flip
    # get the sign-flip vector for every label
    label_flip = list()
    for label, vertidx in zip(labels, label_vertidx):
        if label.hemi == 'both':
            raise ValueError('BiHemiLabel not supported when using sign-flip')
        if vertidx is not None:
            flip = label_sign_flip(label, src)[:, None]
        else:
            flip = None
        label_flip.append(flip)

    return label_flip


def _gen_extract_label_time_course(stcs, labels, src, mode='mean',
                                   allow_empty=False, verbose=None):
    """Generate extract_label_time_course."""
    # if src is a mixed src space, the first 2 src spaces are surf type and
    # the other ones are vol type. For mixed source space n_labels will be the
    # given by the number of ROIs of the cortical parcellation plus the number
    # of vol src space

    if len(src) > 2:
        if src[0]['type'] != 'surf' or src[1]['type'] != 'surf':
            raise ValueError('The first 2 source spaces have to be surf type')
        if any(np.any(s['type'] != 'vol') for s in src[2:]):
            raise ValueError('source spaces have to be of vol type')

        n_aparc = len(labels)
        n_aseg = len(src[2:])
        n_labels = n_aparc + n_aseg
    else:
        n_labels = len(labels)

    # get vertices from source space, they have to be the same as in the stcs
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]

    # do the initialization
    label_vertidx = list()
    for label in labels:
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == 'lh':
                this_vertno = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertno)
            elif slabel.hemi == 'rh':
                this_vertno = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        if len(this_vertidx) == 0:
            msg = ('source space does not contain any vertices for label %s'
                   % label.name)
            if not allow_empty:
                raise ValueError(msg)
            else:
                warn_(msg + '. Assigning all-zero time series to label.')
            this_vertidx = None  # to later check if label is empty

        label_vertidx.append(this_vertidx)

    # mode-dependent initialization
    if mode == 'mean':
        pass  # we have this here to catch invalid values for mode
    elif mode == 'mean_flip':
        # get the sign-flip vector for every label
        label_flip = _get_label_flip(labels, label_vertidx, src[:2])
    elif mode == 'pca_flip':
        # get the sign-flip vector for every label
        label_flip = _get_label_flip(labels, label_vertidx, src[:2])
    elif mode == 'pca_flip_mean':
        # get the sign-flip vector for every label
        label_flip = _get_label_flip(labels, label_vertidx, src[:2])
    elif mode == 'max':
        pass  # we calculate the maximum value later
    else:
        raise ValueError('%s is an invalid mode' % mode)

    # loop through source estimates and extract time series
    for stc in stcs:
        # make sure the stc is compatible with the source space
        for i in range(len(src)):
            if len(stc.vertices[i]) != nvert[i]:
                raise ValueError('stc not compatible with source space. '
                                 'stc has %s time series but there are %s '
                                 'vertices in source space'
                                 % (len(stc.vertices[i]), nvert[i]))

        if any(np.any(svn != vn) for svn, vn in zip(stc.vertices, vertno)):
            raise ValueError('stc not compatible with source space')
        if sum(nvert) != stc.shape[0]:
            raise ValueError('stc not compatible with source space. '
                             'stc has %s vertices but the source space '
                             'has %s vertices'
                             % (stc.shape[0], sum(nvert)))

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        # do the extraction
        label_tc = np.zeros((n_labels, stc.data.shape[1]),
                            dtype=stc.data.dtype)
        if mode == 'mean':
            for i, vertidx in enumerate(label_vertidx):
                if vertidx is not None:
                    label_tc[i] = np.mean(stc.data[vertidx, :], axis=0)
        elif mode == 'mean_flip':
            for i, (vertidx, flip) in enumerate(zip(label_vertidx,
                                                    label_flip)):
                if vertidx is not None:
                    label_tc[i] = np.mean(flip * stc.data[vertidx, :], axis=0)
        elif mode == 'pca_flip':
            for i, (vertidx, flip) in enumerate(zip(label_vertidx,
                                                    label_flip)):
                if vertidx is not None:
                    U, s, V = linalg.svd(stc.data[vertidx, :],
                                         full_matrices=False)
                    # determine sign-flip
                    sign = np.sign(np.dot(U[:, 0], flip))

                    # use average power in label for scaling
                    scale = linalg.norm(s) / np.sqrt(len(vertidx))

                    label_tc[i] = sign * scale * V[0]
        elif mode == 'pca_flip_mean':
            for i, (vertidx, flip) in enumerate(zip(label_vertidx,
                                                    label_flip)):
                if vertidx is not None:
                    U, s, V = linalg.svd(stc.data[vertidx, :],
                                         full_matrices=False)
                    # determine sign-flip
                    flip_refrence = V[0]
                    flip = np.sign(np.corrcoef(
                        flip_refrence, stc.data[vertidx, :]))[1:, 0]
                    label_tc[i] = np.median(
                        flip[:, np.newaxis] * stc.data[vertidx, :], axis=0)
        elif mode == 'max':
            for i, vertidx in enumerate(label_vertidx):
                if vertidx is not None:
                    label_tc[i] = np.max(np.abs(stc.data[vertidx, :]), axis=0)
        else:
            raise ValueError('%s is an invalid mode' % mode)

        # extract label time series for the vol src space
        if len(src) > 2:
            v1 = nvert[0] + nvert[1]
            for i, nv in enumerate(nvert[2:]):

                v2 = v1 + nv
                v = range(v1, v2)
                if nv != 0:
                    label_tc[n_aparc + i] = np.mean(stc.data[v, :], axis=0)

                v1 = v2

        # this is a generator!
        yield label_tc


def extract_label_time_course(stcs, labels, src, mode='mean_flip',
                              allow_empty=False, return_generator=False,
                              verbose=None):
    """Extract label time course for lists of labels and source estimates.

    This function will extract one time course for each label and source
    estimate. The way the time courses are extracted depends on the mode
    parameter.

    Valid values for mode are:

        - 'mean': Average within each label.
        - 'mean_flip': Average within each label with sign flip depending
          on source orientation.
        - 'pca_flip': Apply an SVD to the time courses within each label
          and use the scaled and sign-flipped first right-singular vector
          as the label time course. The scaling is performed such that the
          power of the label time course is the same as the average
          per-vertex time course power within the label. The sign of the
          resulting time course is adjusted by multiplying it with
          "sign(dot(u, flip))" where u is the first left-singular vector,
          and flip is a sing-flip vector based on the vertex normals. This
          procedure assures that the phase does not randomly change by 180
          degrees from one stc to the next.
        - 'max': Max value within each label.


    Parameters
    ----------
    stcs : SourceEstimate | list (or generator) of SourceEstimate
        The source estimates from which to extract the time course.
    labels : Label | list of Label
        The labels for which to extract the time course.
    src : list
        Source spaces for left and right hemisphere.
    mode : str
        Extraction mode, see explanation above.
    allow_empty : bool
        Instead of emitting an error, return all-zero time courses for labels
        that do not have any vertices in the source estimate.
    return_generator : bool
        If True, a generator instead of a list is returned.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose` and
        :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    label_tc : array | list (or generator) of array, shape=(len(labels), n_times)
        Extracted time course for each label and source estimate.
    """  # noqa: E501
    # convert inputs to lists
    if isinstance(stcs, SourceEstimate):
        stcs = [stcs]
        return_several = False
        return_generator = False
    else:
        return_several = True

    if not isinstance(labels, list):
        labels = [labels]

    label_tc = _gen_extract_label_time_course(stcs, labels, src, mode=mode,
                                              allow_empty=allow_empty)

    if not return_generator:
        # do the extraction and return a list
        label_tc = list(label_tc)

    if not return_several:
        # input was a single SoureEstimate, return single array
        label_tc = label_tc[0]

    return label_tc


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
            tc = extract_label_time_course(
                stc, label, src, mode=label_mode)
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
