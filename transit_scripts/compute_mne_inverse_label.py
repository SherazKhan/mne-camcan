cd github/mne-camcan/transit_scripts/
import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne

from utils import (
    decimate_raw, make_surrogates_empty_room, compute_inverse_raw_label)

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

labels = [ll for ll in labels if 'unknown' not in ll.name]
for label in labels:
    label.morph(subject_to=subject, subjects_dir=subjects_dir)
raw.info['sfreq']
decimate_raw(raw, decim=9)

raw.filter(.1, 100, l_trans_bandwidth=0.05, h_trans_bandwidth=1,
           filter_length='auto', phase='zero', fir_window='hann')
import matplotlib.pyplot as plt
%matplotlib inline

out.data.shape
import numpy as np

from sklearn.decomposition import PCA

s_list = list()
for label in labels:
    out = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0, method='MNE', label=label, pick_ori='normal', stop=10000)
    pca = PCA()
    pca.fit(out)
    break
    # U, s, V = np.linalg.svd(out.data, full_matrices=False)
    s_list.append(s)

s_list = np.array(s_list)

fig, axes = plt.subplots(1, 1)
for s in s_list:
    axes.plot(s, alpha=0.01, color='black')
axes.set_xlim(-1, 50)
raws_label = dict()
for method in ['pca_flip_mean', 'mean_flip', 'pca_flip']:
    raws_label[method] = compute_inverse_raw_label(
        raw, labels, inverse_operator, label_mode=method, step=5000)

l.get_vertices_used(np.arange(10242))
[l.get_vertices_used() for l in labels]

labels[22]
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
ax = axes[0]
ax.plot(
    raws_label['pca_flip_mean'].get_data().ravel()[::1000],
    raws_label['mean_flip'].get_data().ravel()[::1000], marker='o',
    linestyle='None', mec='None', alpha=0.1,
    color='red')
ax.set_xlabel('pca flip mean')
ax.set_ylabel('mean flip')

ax = axes[1]
ax.plot(
    raws_label['pca_flip_mean'].get_data().ravel()[::1000],
    raws_label['pca_flip'].get_data().ravel()[::1000], marker='o',
    linestyle='None', mec='None', alpha=0.1, color='violet')
ax.set_xlabel('pca flip mean')
ax.set_ylabel('pca flip')

ax = axes[2]
plt.plot(
    raws_label['pca_flip'].get_data().ravel()[::1000],
    raws_label['mean_flip'].get_data().ravel()[::1000], marker='o',
    linestyle='None', mec='None', alpha=0.1, color='orange')
ax.set_xlabel('pca flip')
ax.set_ylabel('mean flip')

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
colors = 'red', 'orange', 'purple'
for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
    ax = axes[ii]
    ax.plot(raws_label[method].times,
            raws_label[method].get_data().mean(0), color=colors[ii])
    ax.set_xlabel('time [s]')
    ax.set_title(method.replace('_', ' '))
    ax.set_ylim(-1. * 1e-11, 1.5 * 1e-11)
axes[0].set_ylabel('mean(label) [nA.]')

# import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
colors = 'red', 'orange', 'purple'
for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
    ax = axes[ii]
    cov = np.corrcoef(raws_label[method].get_data())
    ax.plot(np.log10(np.linalg.svd(cov, full_matrices=False)[1]), color=colors[ii])

    # im = ax.matshow(cov, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, interpolation='None')
    # # ax.matshow(cov, cmap='RdBu_r', origin='lower', vmin=0.9 * cov.min(), vmax=0.9 * cov.max(), interpolation='nearest')
    ax.set_xlabel('labels')
    ax.set_ylabel('log10(Singular Values)')
    # ax.xaxis.set_ticks_position('bottom')
    ax.set_title(method.replace('_', ' '))


fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
colors = 'red', 'orange', 'purple'
ticks = [0.1, 1, 10, 100]
# raws_label[method].get_data().shape
for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
    ax = axes[ii]
    psd, freqs = mne.time_frequency.psd_array_welch(
        raws_label[method].get_data(), n_fft=8192, n_overlap=4096,
        sfreq=raw.info['sfreq'], fmin=0.1, fmax=100)
    psd = np.log10(psd ** 2)
    log_freqs = np.log10(freqs)
    ax.plot(log_freqs, psd.T, color='black', alpha=0.1)
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels(ticks)
    ax.set_title(method.replace('_', ' '))
    ax.set_xlabel('Frequency [Hz]')
    ax.grid(True)
axes[0].set_ylabel('log10(x ** 2) [Am]')


raw_label.rename_channels(
    dict(zip(raw_label.ch_names,
             ['l%03d' % ii for ii, _ in enumerate(raw_label.ch_names)])))

# raw_label.filter(1, 100, picks=list(range(448)))
plt.plot(raw_label.times, raw_label.get_data().mean(0))

fig = raw_label.plot_psd(fmin=2, fmax=30, n_fft=8912, picks=list(range(448)), average=False, area_alpha=0.01, n_overlap=4096);


raw_label.save('labels_aparc_sk_broadband_raw.fif', overwrite=True)

del raw_label

raw_noise = mne.io.read_raw_fif(raw_noise_fname)
raw_noise.rename_channels(
    dict(zip(raw_noise.ch_names,
             mne.utils._clean_names(raw_noise.ch_names))))

decimate_raw(raw_noise, decim=24)
raw_noise.preload = True
covs = list()
for fmin, fmax in [(2, 4), (8, 12), (30, 50), (0, 100)]:
    raw_cur = raw_noise.copy()
    raw_cur.filter(fmin, fmax)
    covs.append(mne.compute_raw_covariance(raw_cur))

from mne.report import Report
rep = Report('covs')
cov = covs[0]
cov.plot
for ii, (fmin, fmax) in enumerate([(2, 4), (8, 12), (30, 50), (0, 100)]):
    figs = covs[ii].plot(raw_noise.info, show_svd=False, colorbar=True)
    # rep.add_figs_to_section(figs, section='%i-%i' % (fmin, fmax),
    # captions=['cov', 'eigen'])

import numpy as np
np.angle

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
decimate_raw(raw_noise, decim=24)
raw_repro = make_surrogates_empty_room(raw_noise, fwd_fixed, inverse_operator)

raw_label_noise = compute_inverse_raw_label(
    raw_repro, labels, inverse_operator, label_mode='pca_flip_mean')

raw_label_noise.save('labels_aparc_sk_broadband_noise_raw.fif', overwrite=True)
