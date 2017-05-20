# cd transit_scripts/
import glob
import os.path as op
from mne.datasets.brainstorm import bst_resting
import mne

from utils import (
    decimate_raw, make_surrogates_empty_room, compute_inverse_raw_label)

mne.utils.set_log_level('warning')

data_path = bst_resting.data_path()

raw_noise_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_noise_20111104_02_comp_raw.fif')

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

spacing = 'ico5'

noise_cov_fname = op.join(
    meg_dir, subject, '%s-%s-cov.fif' % (subject, spacing))
fwd_fname = op.join(meg_dir, subject, '%s_%s-fwd.fif' % (subject, spacing))

fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']
noise_cov = mne.read_cov(noise_cov_fname)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov)

labels = [mne.read_label(fpath) for fpath in glob.glob('./testing_labels/*label')]

raw = mne.io.read_raw_fif('brainstorm_testing_rest_raw.fif')
raw.load_data()
raw.pick_types(meg=True, eeg=False)
raw.filter(0.05, 100, l_trans_bandwidth=0.05, h_trans_bandwidth=1,
           fir_design='firwin', filter_length='auto', phase='zero', fir_window='hann')

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(raw.get_data().mean(0))
raws_label = dict()
methods = ['pca_flip_mean',
           'mean_flip',
           'pca_flip',
           ('pca_flip_truncated', .9),
           ('pca_flip_truncated', 3)]

for method in methods:
    raws_label[method] = compute_inverse_raw_label(
        raw, labels, inverse_operator, label_mode=method)

event_id = 1
event_overlap = 8
event_length = 30
raw_length = (raw.last_samp - raw.first_samp) / raw.info['sfreq']

events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap,
                                      start=0, stop=raw_length-event_length)

fig, axes = plt.subplots(len(methods), 1, sharey=True, sharex=True, figsize=(10, 8))
axes = axes.ravel()
ax = axes[ii]
# x = raws_label[method].get_data().mean(0)
if isinstance(method, tuple):
    label = method[0] + str(method[1])
else:
    label = method
    
colors = plt.cm.Set2(np.arange(len(methods)))
X_psds_raws = dict()
for ii, method in enumerate(methods):

    psds, freqs = mne.time_frequency.psd_welch(
        raws_label[method],
        picks=list(range(448)), n_fft=4096, n_overlap=4096/2, fmin=0, fmax=150)

    X_psd = 10 * np.log10(psds ** 2)
    X_psds_raws[method] = (X_psd, freqs)

X_psds_epochs = dict()
for ii, method in enumerate(methods):
    epochs = mne.Epochs(
        raws_label[method], events, event_id, tmin=0,
        tmax=event_length, baseline=None, preload=True, proj=False, reject=None,
        picks=list(range(448)))

    psds, freqs = mne.time_frequency.psd_welch(
        epochs, picks=list(range(448)), n_fft=4096, n_overlap=4096 / 2,
        fmin=0, fmax=150)

    X_psd = 10 * np.log10(psds ** 2).mean(0)
    X_psds_epochs[method] = (X_psd, freqs)


label_colors = {ll.name: ll.color for ll in mne.read_labels_from_annot(
                parc='aparc_sk', subject='fsaverage', subjects_dir=subjects_dir)
                if not 'known' in ll.name}


def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,
         tick labels, axis labels, title and legend
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)


for method in methods:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, sharex=True)
    fig.patch.set_facecolor('black')
    ax1, ax2 = axes.ravel()
    ax1.set_axis_bgcolor('black')
    ax2.set_axis_bgcolor('black')
    set_foregroundcolor(ax1, 'white')
    set_foregroundcolor(ax2, 'white')
    X_psd, freqs = X_psds_epochs[method]
    for ii, label in enumerate(labels):
        color = label_colors[label.name]
        ax1.plot(freqs, X_psd[ii], color=color, alpha=0.3)
    X_psd, freqs = X_psds_raws[method]
    for ii, label in enumerate(labels):
        color = label_colors[label.name]
        ax2.semilogx(freqs, X_psd[ii], color=color, alpha=0.3)
    ax1.set_title('epochs')
    ax2.set_title('raw')
    fig.suptitle(method, color='white')
    if isinstance(method, tuple):
        method_ = method[0] + str(method[1])
    else:
        method_ = method
    ax1.set_ylim(-550, -400)
    ax2.set_ylim(-550, -400)
    fig.savefig('epochs_raw_comp_%s.png' % method_, dpi=300,
                facecolor=fig.get_facecolor(), edgecolor='none')

from scipy import linalg
cov_dict = {method: np.cov(raws_label[method].get_data()) for method in methods}
plt.figure(figsize=(8, 6))
colors = plt.cm.Set2(np.arange(len(methods)))
for ii, method in enumerate(methods):
    cov = cov_dict[method]
    U, s, V = linalg.svd(cov, full_matrices=False)
    if isinstance(method, tuple):
        method_ = method[0] + str(method[1])
    else:
        method_ = method
    plt.plot(np.log10(s), label=method_, color=colors[ii], linewidth=3)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Labels')
plt.ylabel('log10(s)')
plt.savefig('eigen_spectra_methods.png', dpi=300)


plt.figure(figsize=(8, 6))
colors = plt.cm.Set2(np.arange(len(methods)))
for ii, method in enumerate(methods):
    x = raws_label[method].get_data().mean(0)
    if isinstance(method, tuple):
        method_ = method[0] + str(method[1])
    else:
        method_ = method
    plt.psd(x, Fs=raw.info['sfreq'], NFFT=8196, label=method_, color=colors[ii], linewidth=3)
plt.grid(True)
plt.legend(loc='best')
plt.savefig('mean_spectra_methods.png', dpi=300)

# X_psds_epochs_sk = dict()
# for ii, method in enumerate(methods):
#     epochs = mne.Epochs(
#         raw, events, event_id, tmin=0,
#         tmax=event_length, baseline=None, preload=True, proj=False, reject=None)
# 
#     raw_epochs = mne.io.RawArray(
#         np.hstack(epochs.get_data()), info=epochs.info)
# 
#     raw_label = compute_inverse_raw_label(
#         raw, labels, inverse_operator, label_mode=method)
# 
#     psds, freqs = mne.time_frequency.psd_welch(
#         raw_label,
#         picks=list(range(448)), n_fft=4096, n_overlap=4096/2, fmin=0, fmax=150)
# 
#     X_psd = 10 * np.log10(psds ** 2).mean(0)
#     X_psds_epochs_sk[method] = (X_psd, freqs)

# 
#     for ii, label in enumerate(labels):
#         ax.semilogx(freqs, X_psd[ii], color=label.color, alpha=0.3)
#     ax.set_title(label)
#     ax.set_ylim(-270, -200)
#     # ax.set_ylim(-270, -200)
# 
# 
# plt.figure()
# for ii, label in enumerate(labels):
#     plt.semilogx(freqs, X_psd[ii], color=label.color, alpha=0.3)


mne.viz.plot_events(events, sfreq=raw.info['sfreq']);

raw.times[events[:, 0]]
plt.figure()
plt.plot(raw.times[events][:,0], events[:, 0], marker='+', linestyle='None')


fig, axes = plt.subplots(len(methods), 1, sharey=True, sharex=True, figsize=(10, 8))
axes = axes.ravel()
colors = plt.cm.Set2(np.arange(len(methods)))
for ii, method in enumerate(methods):
    ax = axes[ii]
    x = raws_label[method].get_data().ravel()[:500]
    if isinstance(method, tuple):
        label = method[0] + str(method[1])
    else:
        label = method
    ax.plot(x, color=colors[ii])
    ax.set_title(label)
    ax.grid(True)

import numpy as np
np.corrcoef([raws_label[method].get_data().ravel()[::1000] for method in methods])


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

# 
# import matplotlib.pyplot as plt
# %matplotlib inline
# 
# out.data.shape
# import numpy as np
# 
# from sklearn.decomposition import PCA
# 
# s_list = list()
# for label in labels:
#     out = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0, method='MNE', label=label, pick_ori='normal', stop=10000)
#     pca = PCA()
#     pca.fit(out)
#     break
#     # U, s, V = np.linalg.svd(out.data, full_matrices=False)
#     s_list.append(s)
# 
# s_list = np.array(s_list)
# 
# fig, axes = plt.subplots(1, 1)
# for s in s_list:
#     axes.plot(s, alpha=0.01, color='black')
# axes.set_xlim(-1, 50)
# raws_label = dict()
# for method in ['pca_flip_mean', 'mean_flip', 'pca_flip']:
#     raws_label[method] = compute_inverse_raw_label(
#         raw, labels, inverse_operator, label_mode=method, step=5000)
# 
# l.get_vertices_used(np.arange(10242))
# [l.get_vertices_used() for l in labels]
# 
# labels[22]
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# ax = axes[0]
# ax.plot(
#     raws_label['pca_flip_mean'].get_data().ravel()[::1000],
#     raws_label['mean_flip'].get_data().ravel()[::1000], marker='o',
#     linestyle='None', mec='None', alpha=0.1,
#     color='red')
# ax.set_xlabel('pca flip mean')
# ax.set_ylabel('mean flip')
# 
# ax = axes[1]
# ax.plot(
#     raws_label['pca_flip_mean'].get_data().ravel()[::1000],
#     raws_label['pca_flip'].get_data().ravel()[::1000], marker='o',
#     linestyle='None', mec='None', alpha=0.1, color='violet')
# ax.set_xlabel('pca flip mean')
# ax.set_ylabel('pca flip')
# 
# ax = axes[2]
# plt.plot(
#     raws_label['pca_flip'].get_data().ravel()[::1000],
#     raws_label['mean_flip'].get_data().ravel()[::1000], marker='o',
#     linestyle='None', mec='None', alpha=0.1, color='orange')
# ax.set_xlabel('pca flip')
# ax.set_ylabel('mean flip')
# 
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# colors = 'red', 'orange', 'purple'
# for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
#     ax = axes[ii]
#     ax.plot(raws_label[method].times,
#             raws_label[method].get_data().mean(0), color=colors[ii])
#     ax.set_xlabel('time [s]')
#     ax.set_title(method.replace('_', ' '))
#     ax.set_ylim(-1. * 1e-11, 1.5 * 1e-11)
# axes[0].set_ylabel('mean(label) [nA.]')
# 
# # import numpy as np
# 
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# colors = 'red', 'orange', 'purple'
# for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
#     ax = axes[ii]
#     cov = np.corrcoef(raws_label[method].get_data())
#     ax.plot(np.log10(np.linalg.svd(cov, full_matrices=False)[1]), color=colors[ii])
# 
#     # im = ax.matshow(cov, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, interpolation='None')
#     # # ax.matshow(cov, cmap='RdBu_r', origin='lower', vmin=0.9 * cov.min(), vmax=0.9 * cov.max(), interpolation='nearest')
#     ax.set_xlabel('labels')
#     ax.set_ylabel('log10(Singular Values)')
#     # ax.xaxis.set_ticks_position('bottom')
#     ax.set_title(method.replace('_', ' '))
# 
# 
# fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# colors = 'red', 'orange', 'purple'
# ticks = [0.1, 1, 10, 100]
# # raws_label[method].get_data().shape
# for ii, method in enumerate(['pca_flip_mean', 'mean_flip', 'pca_flip']):
#     ax = axes[ii]
#     psd, freqs = mne.time_frequency.psd_array_welch(
#         raws_label[method].get_data(), n_fft=8192, n_overlap=4096,
#         sfreq=raw.info['sfreq'], fmin=0.1, fmax=100)
#     psd = np.log10(psd ** 2)
#     log_freqs = np.log10(freqs)
#     ax.plot(log_freqs, psd.T, color='black', alpha=0.1)
#     ax.set_xticks(np.log10(ticks))
#     ax.set_xticklabels(ticks)
#     ax.set_title(method.replace('_', ' '))
#     ax.set_xlabel('Frequency [Hz]')
#     ax.grid(True)
# axes[0].set_ylabel('log10(x ** 2) [Am]')
# 
# 
# raw_label.rename_channels(
#     dict(zip(raw_label.ch_names,
#              ['l%03d' % ii for ii, _ in enumerate(raw_label.ch_names)])))
# 
# # raw_label.filter(1, 100, picks=list(range(448)))
# plt.plot(raw_label.times, raw_label.get_data().mean(0))
# 
# fig = raw_label.plot_psd(fmin=2, fmax=30, n_fft=8912, picks=list(range(448)), average=False, area_alpha=0.01, n_overlap=4096);
# 
# 
# raw_label.save('labels_aparc_sk_broadband_raw.fif', overwrite=True)
# 
# del raw_label
# 
# raw_noise = mne.io.read_raw_fif(raw_noise_fname)
# raw_noise.rename_channels(
#     dict(zip(raw_noise.ch_names,
#              mne.utils._clean_names(raw_noise.ch_names))))
# 
# decimate_raw(raw_noise, decim=24)
# raw_noise.preload = True
# covs = list()
# for fmin, fmax in [(2, 4), (8, 12), (30, 50), (0, 100)]:
#     raw_cur = raw_noise.copy()
#     raw_cur.filter(fmin, fmax)
#     covs.append(mne.compute_raw_covariance(raw_cur))
# 
# from mne.report import Report
# rep = Report('covs')
# cov = covs[0]
# cov.plot
# for ii, (fmin, fmax) in enumerate([(2, 4), (8, 12), (30, 50), (0, 100)]):
#     figs = covs[ii].plot(raw_noise.info, show_svd=False, colorbar=True)
#     # rep.add_figs_to_section(figs, section='%i-%i' % (fmin, fmax),
#     # captions=['cov', 'eigen'])
# 
# import numpy as np
# np.angle
# 
# fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
# decimate_raw(raw_noise, decim=24)
# raw_repro = make_surrogates_empty_room(raw_noise, fwd_fixed, inverse_operator)
# 
# raw_label_noise = compute_inverse_raw_label(
#     raw_repro, labels, inverse_operator, label_mode='pca_flip_mean')
# 
# raw_label_noise.save('labels_aparc_sk_broadband_noise_raw.fif', overwrite=True)
