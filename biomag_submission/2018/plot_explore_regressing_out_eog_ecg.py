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

meg_inp_dir = op.join(data_path, 'passive', )
meg_dir = op.join(data_path, 'meg_dir')

subjects = ['CC110033', 'CC110037', 'CC110045']


def _regress_out_confounds(data_fit, data_apply, data_conf_fit,
                           data_conf_apply,
                           mode='multi',
                           estimator=None):
    if estimator is None:
        est = make_pipeline(
            RobustScaler(),
            LinearRegression())
    else:
        est = clone(estimator)

    Y_pred = np.empty(data_apply.shape)
    if mode == 'single':
        for ii in range(data_fit.shape[0]):
            est.fit(data_conf_fit.T, data_fit[ii])
            Y_pred[ii] = est.predict(data_conf_apply.T).ravel()
    elif mode == 'multi':
        est.fit(data_conf_fit.T, data_fit.T)
        Y_pred[:] = est.predict(data_conf_apply.T).T
    else:
        raise ValueError('Noo!')

    data_clean = data_apply - Y_pred
    return data_clean


def _regress_out_ecg_eog(raw, reject, decim=None, mode='epochs'):
    if 'ecg' not in raw and 'eog' not in raw:
        raise ValueError('There is neither EOG nor ECG.')
    picks_meeg = mne.pick_types(raw.info, meg=True, eeg=True)
    picks_ecg = mne.pick_types(raw.info, meg=False,
                               eog=False, ecg=True)
    picks_eog = mne.pick_types(raw.info, meg=False,
                               eog=True, ecg=False)
    data_apply = raw.get_data().copy()
    if mode == 'epochs':
        if len(picks_eog) > 0:
            data_fit = mne.preprocessing.create_eog_epochs(
                raw, reject=reject).get_data()
            data_fit = np.concatenate(data_fit, axis=-1)
            data_clean = _regress_out_confounds(
                data_conf_fit=data_fit[picks_eog][:, ::decim],
                data_fit=data_fit[picks_meeg][:, ::decim],
                data_conf_apply=data_apply[picks_eog],
                data_apply=data_apply[picks_meeg],
                estimator=None)
        else:
            data_clean = data_apply[picks_meeg]

        if len(picks_ecg) > 0:
            data_fit = mne.preprocessing.create_ecg_epochs(
                raw, reject=reject).filter(8, 16, picks=picks_ecg).get_data()
            data_fit = np.concatenate(data_fit, axis=-1)

            data_clean = _regress_out_confounds(
                data_conf_fit=data_fit[picks_ecg][:, ::decim],
                data_fit=data_fit[picks_meeg][:, ::decim],
                data_conf_apply=data_apply[picks_ecg],
                data_apply=data_clean,
                estimator=None)

    elif mode == 'raw':
        picks_conf = np.concatenate([picks_eog, picks_ecg], axis=0)
        data_fit, _ = _reject_data_segments(
            data_apply, reject=reject, flat=None, decim=decim, info=raw.info,
            tstep=2.)

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


def _process_subject(subject):
    raw = mne.io.read_raw_fif(
        op.join(meg_inp_dir, 'sub-{}'.format(subject),
                'meg', 'passive_raw.fif'))

    raw.load_data()
    raw.filter(1, 100)

    ave_ecg = mne.preprocessing.create_ecg_epochs(raw).average()

    ave_eog = mne.preprocessing.create_eog_epochs(raw).average()

    reject = get_rejection_threshold(mne.preprocessing.create_eog_epochs(raw))

    raw_clean = _regress_out_ecg_eog(raw, reject=reject, decim=8)

    ave_eog = mne.preprocessing.create_eog_epochs(
        raw, reject=reject).average()
    ave_eog_after = mne.preprocessing.create_eog_epochs(
        raw_clean, reject=reject).average()
    ave_ecg = mne.preprocessing.create_ecg_epochs(
        raw, reject=reject).average()
    ave_ecg_after = mne.preprocessing.create_ecg_epochs(
        raw_clean, reject=reject).average()
    return subject, ave_ecg, ave_ecg_after, ave_eog, ave_eog_after


out = Parallel(n_jobs=8)(delayed(_process_subject)(
    subject=subject) for subject in subjects)

for subject, ave_ecg, ave_ecg_after, ave_eog, ave_eog_after in out:

    plot_overlay_evoked(ave_eog, ave_eog_after, title='EOG', show=True)
    plt.suptitle('EOG artifact before (red) and after regression '
                 '(black [{}]'.format(subject))
    plt.savefig('./figures_qc/{}-reg-eog.pdf'.format(subject))

    plot_overlay_evoked(ave_ecg, ave_ecg_after, title='ECG', show=True)
    plt.suptitle('ECG artifact before (red) and after regression '
                 '(black) [{}]'.format(subject))
    plt.savefig('./figures_qc/{}-reg-ecg.pdf'.format(subject))
