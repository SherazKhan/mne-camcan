import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import mne

from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mne.utils import _reject_data_segments
from mne.viz.ica import _plot_ica_overlay_evoked as plot_overlay_evoked
from autoreject import get_rejection_threshold

data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')

meg_dir = op.join(data_path, 'meg_dir')
kinds = ['rest', 'task']
subjects = ['CC110033', 'CC110037', 'CC110045']


def _run_maxfilter(raw, coord_frame='head'):
    cal = op.join('../../camcan/library/sss_cal.dat')
    ctc = op.join('../../camcan/library/ct_sparse.fif')
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=cal,
        cross_talk=ctc,
        st_duration=10.,
        st_correlation=.98,
        coord_frame=coord_frame)
    return raw


def _regress_out_confounds(data_fit, data_apply, data_conf_fit,
                           data_conf_apply,
                           mode='multi-output',
                           estimator=None):
    if estimator is None:
        est = make_pipeline(
            RobustScaler(),
            LinearRegression())
    else:
        est = clone(estimator)

    Y_pred = np.empty(data_apply.shape)
    if mode == 'single-output':
        for ii in range(data_fit.shape[0]):
            est.fit(data_conf_fit.T, data_fit[ii])
            Y_pred[ii] = est.predict(data_conf_apply.T).ravel()
    elif mode == 'multi-output':
        est.fit(data_conf_fit.T, data_fit.T)
        Y_pred[:] = est.predict(data_conf_apply.T).T
    else:
        raise ValueError('Noo!')

    data_clean = data_apply - Y_pred
    return data_clean


def _regress_out_ecg_eog(
        raw, reject_eog=None, reject_ecg=None, decim=None, mode='epochs'):
    if 'ecg' not in raw and 'eog' not in raw:
        raise ValueError('There is neither EOG nor ECG.')
    picks_meeg = mne.pick_types(raw.info, meg=True, eeg=True)
    picks_ecg = mne.pick_types(raw.info, meg=False, eeg=False,
                               eog=False, ecg=True)
    picks_eog = mne.pick_types(raw.info, meg=False, eeg=False,
                               eog=True, ecg=False)
    data_apply = raw.get_data().copy()
    if mode == 'epochs':
        if len(picks_eog) > 0:
            data_fit = mne.preprocessing.create_eog_epochs(
                raw, reject=reject_eog).get_data()
            print('Using %s eog epochs for fitting.' % len(data_fit))
            if len(data_fit) >= 5:
                data_fit = np.concatenate(data_fit, axis=-1)
                data_clean = _regress_out_confounds(
                    data_conf_fit=data_fit[picks_eog][:, ::decim],
                    data_fit=data_fit[picks_meeg][:, ::decim],
                    data_conf_apply=data_apply[picks_eog],
                    data_apply=data_apply[picks_meeg],
                    estimator=None)
            else:
                data_clean = None
        else:
            data_clean = data_apply[picks_meeg]

        if len(picks_ecg) > 0:
            data_fit = (mne.preprocessing.create_ecg_epochs(
                raw, reject=reject_ecg)
                # .filter(8, 16, picks=picks_ecg)
                .get_data())
            print('Using %s ecg epochs for fitting.' % len(data_fit))
            if data_clean is None:
                data_clean = data_apply[picks_meeg]
            if len(data_fit) >= 5:
                data_fit = np.concatenate(data_fit, axis=-1)

                data_clean = _regress_out_confounds(
                    data_conf_fit=data_fit[picks_ecg][:, ::decim],
                    data_fit=data_fit[picks_meeg][:, ::decim],
                    data_conf_apply=data_apply[picks_ecg],
                    data_apply=data_clean,
                    estimator=None)
            else:
                raise ValueError('Could not find sufficiently many samples to '
                                 'preform EOG/ECG regression')

    elif mode == 'raw':
        # XXX disambiguate correctly
        picks_conf = np.concatenate([picks_eog, picks_ecg], axis=0)
        data_fit, _ = _reject_data_segments(
            data_apply, reject=reject_ecg, flat=None, decim=decim,
            info=raw.info, tstep=2.)

        if data_clean is None:
            data_clean = data_apply
        else:
            data_apply = data_clean

        data_clean = _regress_out_confounds(
            data_conf_fit=data_fit[picks_conf],
            data_fit=data_fit[picks_meeg],
            data_conf_apply=data_apply[picks_conf],
            data_apply=data_apply[picks_meeg],
            estimator=None)

    data_apply[picks_meeg] = data_clean
    raw_clean = mne.io.RawArray(
        data=data_apply,
        info=raw.info.copy())
    return raw_clean


def _parse_bad_channels(sss_log):
    with open(sss_log) as fid:
        bad_lines = {l for l in fid.readlines() if 'Static bad' in l}
    bad_channels = list()
    for line in bad_lines:
        chans = line.split(':')[1].strip(' \n').split(' ')
        for cc in chans:
            ch_name = 'MEG%01d' % int(cc)
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
    return bad_channels


def _process_subject(subject, kind):
    meg_inp_dir = op.join(data_path, kind)
    raw = mne.io.read_raw_fif(
        op.join(meg_inp_dir, '{}'.format(subject),
                '{}_raw.fif'.format(kind)))

    sss_log = op.join(meg_inp_dir, '{}'.format(subject),
                      'mf2pt2_{}_raw.log'.format(kind))
    raw.info['bads'].extend(_parse_bad_channels(sss_log))
    raw.load_data()
    raw = _run_maxfilter(raw, 'head')

    raw.filter(0.1, 100)
    raw.notch_filter(np.arange(50, 251, 50))

    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs)
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs)
    else:
        reject_eog = None

    if reject_eog is None:
        reject_eog = reject_ecg
    if reject_ecg is None:
        reject_ecg = reject_eog

    raw_clean = _regress_out_ecg_eog(
        raw, reject_eog=reject_eog, reject_ecg=reject_ecg, decim=8)

    ave_ecg = ecg_epochs.average()
    ave_eog = eog_epochs.average()

    ave_eog_after = mne.preprocessing.create_eog_epochs(
        raw_clean, reject=reject_eog).average()
    ave_ecg_after = mne.preprocessing.create_ecg_epochs(
        raw_clean, reject=reject_ecg).average()

    return (subject, ave_ecg, ave_ecg_after, ave_eog, ave_eog_after,
            raw_clean, kind)


out = Parallel(n_jobs=2)(delayed(_process_subject)(
    subject=subject, kind=kind) for subject in subjects for kind in kinds)

for (subject, ave_ecg, ave_ecg_after, ave_eog, ave_eog_after,
     raw_clean, kind) in out:

    raw_clean.save(
        op.join(data_path, 'meg_dir', subject, '{}-cln-raw.fif'.format(kind)),
        overwrite=True)

    plot_overlay_evoked(ave_eog, ave_eog_after, title='EOG', show=True)
    plt.suptitle('EOG artifact before (red) and after regression '
                 '(black [{}]'.format(subject))
    plt.savefig('./figures_qc/{}-{}-reg-eog.png'.format(
        subject, kind), dpi=150)

    plot_overlay_evoked(ave_ecg, ave_ecg_after, title='ECG', show=True)
    plt.suptitle('ECG artifact before (red) and after regression '
                 '(black) [{}]'.format(subject))
    plt.savefig('./figures_qc/{}-{}-reg-ecg.png'.format(
        subject, kind), dpi=150)
