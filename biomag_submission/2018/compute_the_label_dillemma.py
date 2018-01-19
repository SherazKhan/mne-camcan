import numpy as np
import os.path as op
from joblib import Parallel, delayed
from utils import decimate_raw
import mne
from mne import minimum_norm as mn
from mne.utils import split_list
from mne.externals.h5io import write_hdf5

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from utils import extract_label_time_course

data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')

meg_dir = op.join(data_path, 'meg_dir')
results_dir = './results'
subjects = ['CC110033', 'CC110037', 'CC110045']
label_modes = ['mean_flip',
               'pca_flip',
               'pca_flip_truncated']

subjects_dir = op.join(data_path, 'subjects')
N_JOBS = 8
DEBUG = False


def _label_morph(label, subject_to, subjects_dir):
    return label.morph(subject_to=subject_to, subjects_dir=subjects_dir)


def _get_label_props(parc, src, fwd, subject, debug=False):
    labels = mne.read_labels_from_annot(
        parc=parc, subject='fsaverage', subjects_dir=subjects_dir)

    if debug:
        labels = labels[10:12]

    labels = Parallel(n_jobs=N_JOBS)(delayed(_label_morph)(
        ll, subject_to=subject, subjects_dir=subjects_dir)
        for ll in labels if '???' not in ll.name and 'unknown' not in ll.name)
    mag_map = mne.sensitivity_map(fwd=fwd, ch_type='mag')
    grad_map = mne.sensitivity_map(fwd=fwd, ch_type='grad')
    n_total = float(src[0]['nuse'] + src[1]['nuse'])
    ff = np.array
    props = dict(
        mag_sens_mean=ff([mag_map.in_label(ll).data.mean() for ll in labels]),
        mag_sens_std=ff([mag_map.in_label(ll).data.std() for ll in labels]),
        grad_sens_mean=ff(
            [grad_map.in_label(ll).data.mean() for ll in labels]),
        grad_sens_std=ff([grad_map.in_label(ll).data.std() for ll in labels]),
        relative_size=ff([len(ll.get_vertices_used(
                          src[{'lh': 0, 'rh': 1}[ll.hemi]]['vertno'])) /
                         n_total for ll in labels]))
    return labels, props


def _initialize_data(subject, debug=False):
    fwd_fname = op.join(meg_dir, subject, 'ico5-white-ico5-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)

    inv_fname = op.join(meg_dir, subject, 'mne-er-ico5-NoneHzlp-inv.fif')

    inverse_operator = mn.read_inverse_operator(inv_fname)
    src = inverse_operator['src']

    labels_aparc, labels_aparc_props = _get_label_props(
        'aparc', src, fwd, subject, debug=debug)

    labels_aparc_sk, labels_aparc_sk_props = _get_label_props(
        'aparc_sk', src, fwd, subject, debug=debug)

    labels_hcp, labels_hcp_props = _get_label_props(
        'HCPMMP1', src, fwd, subject, debug=debug)

    annot_names = ['aparc', 'aparc_sk', 'hcp']

    labels_map = dict(
        zip(annot_names,
            [labels_aparc_sk, labels_aparc, labels_hcp]))

    label_props = dict(
        zip(annot_names,
            [labels_aparc_sk_props, labels_aparc_props, labels_hcp_props])
    )

    return labels_map, label_props, src, fwd, inverse_operator


def _get_label_outputs(
        stc, labels, label_modes, src, source_decim, n_comps, return_ts):
    scores = np.empty((len(label_modes), len(labels)))
    n_comps_ = np.empty(len(labels))
    if return_ts:
        tcs = np.empty(
            (len(label_modes), len(labels), len(stc.times[::source_decim])))
    else:
        tcs = None
    for ii_label, label in enumerate(labels):
        print(label.name)
        label_data = stc.in_label(label).data
        pca = PCA()
        pca.fit(label_data.T)
        n_comps_[ii_label] = (
            pca.explained_variance_ratio_.cumsum() <= n_comps).sum()
        for ii_mode, label_mode in enumerate(label_modes):
            tc = extract_label_time_course(
                stc, label, src, mode=label_mode, verbose=0)
            if return_ts:
                tcs[ii_mode, ii_label] = tc[0, ::source_decim]

            mod = LinearRegression()
            mod.fit(tc.T, label_data[:, ::source_decim].T)
            score = mod.score(tc.T, label_data[:, ::source_decim].T)
            scores[ii_mode, ii_label] = score
    return tcs, scores, n_comps_


def _run_label_properties(raw, labels, label_modes, inverse_operator,
                          source_decim=1, step=10000, n_jobs=8,
                          return_ts=False, debug=False):
    if debug:
        print('entring debug mode')
        raw_ = raw.copy().crop(0.0, 1.0)
        step = raw_.times.shape[0]
        labels = labels[:1]
        label_modes = label_modes[:1]
        n_jobs = 1
    else:
        raw_ = raw

    n_source_times = int(raw_.get_data().shape[1] / source_decim)
    index = np.arange(len(raw_.times)).astype(int)

    last = len(index)
    n_comps = .99
    windows = [(start, start + min(step, last - start)) for
               start in index[::step]]

    X_n_comps = np.empty((len(labels), len(windows)), dtype=np.float)

    X_r2 = np.empty(
        (len(label_modes), len(labels)), dtype=np.float)

    if return_ts:
        X_stc = np.empty(
            (len(label_modes), len(labels), n_source_times), dtype=np.float)
    else:
        X_stc = None

    src = inverse_operator['src']
    for i_win, (start, stop) in enumerate(windows):
        stc = mne.minimum_norm.apply_inverse_raw(
            raw_, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        out = Parallel(n_jobs=n_jobs)(delayed(_get_label_outputs)(
            stc, lls, label_modes, src, source_decim, n_comps, return_ts)
            for lls in
            split_list(labels, n_jobs))
        tc, scores, n_comps_ = zip(*out)
        start_target = int(start // source_decim)
        stop_target = int(stop // source_decim)
        X_n_comps[:, i_win] = np.concatenate(n_comps_, 0)
        X_r2[..., i_win] = np.concatenate(scores, 0)
        if return_ts:
            X_stc[..., start_target:stop_target] = np.concatenate(tc, 0)
    return X_n_comps, X_r2, X_stc


def _run_experiment(subject):
    kind = 'rest'
    labels_map, label_props, src, fwd, inverse_operator = _initialize_data(
        subject, debug=DEBUG)
    raw_fname = op.join(meg_dir, subject, '{}-sss-raw.fif'.format(kind))
    for fmin, fmax in ((None, None), (8, 12), (14, 30)):
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.crop(0.0, 5.)
        decimate_raw(raw, 8)
        if fmin is not None and fmax is not None:
            raw.filter(
                fmin, fmax, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                filter_length='auto', phase='zero', fir_window='hann')
        for label_type, labels in labels_map.items():
            out = _run_label_properties(
                raw, labels, label_modes, inverse_operator, return_ts=True,
                debug=DEBUG)
            out_fname = op.join(
                results_dir, '{}-{}-{}-{}-results.h5'.format(
                    subject, label_type, fmin, fmax))
            result = {
                'fmin': fmin, 'fmax': fmax, 'label_type': label_type,
                'subject': subject, 'data_type': 'rest',
                'data': out, 'label_props': label_props}
            write_hdf5(out_fname, result)


for subject in subjects:
    _run_experiment(subject)
