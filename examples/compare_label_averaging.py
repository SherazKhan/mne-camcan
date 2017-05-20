import mne
import os.path as op
import matplotlib.pyplot as plt
from camcan.library.config import ctc, cal
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import glob
import numpy as np
from camcan.utils import stft
plt.ion()


subjects = ['CC110033', 'CC110037', 'CC110045']
subject = subjects[0]

data_path = '/cluster/transcend/sheraz/Dropbox/mne-camcan-data/'
subjects_dir = op.join(data_path,'recons')
subject_dir = op.join(subjects_dir,subject)
bem_dir = op.join(subject_dir,'bem')
trans_file = op.join(data_path, 'trans',subject + '-trans.fif')
labels_fname  = glob.glob(op.join(data_path, 'labels', '*.label'))
labels = [mne.read_label(label, subject='fsaverageSK', color='r')
          for label in labels_fname]
for index, label in enumerate(labels):
    label.values.fill(1.0)
    labels[index] = label

labels = [label.morph('fsaverageSK', subject, subjects_dir=subjects_dir) for label in labels]

event_id = 1
event_overlap = 8
event_length = 30
spacing='ico5'

raw_fname = op.join(
    data_path, 'rest', 'sub-' + subject,'meg', 'rest_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_length = (raw.last_samp-raw.first_samp)/raw.info['sfreq']
raw.info['bads'] +=  [u'MEG2113', u'MEG1941', u'MEG1412', u'MEG2331']

raw = mne.preprocessing.maxwell_filter(raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       origin=(0., 0., 0.04))

projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, eog_events = compute_proj_eog(raw, n_grad=1, n_mag=2)

raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog

raw.apply_proj()

raw.filter(.5, 100, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

reject = dict(grad=4000e-13, mag=4e-12)
events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap,
                                      start=0, stop=raw_length-event_length)
epochs = mne.Epochs(raw, events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
epochs.resample(300.)

bem_fname = op.join(bem_dir, '%s-src.fif' % subject)
src_fname = op.join(bem_dir, '%s-src.fif' % spacing)

bem = mne.read_bem_solution(bem_fname)
src = mne.read_source_spaces(src_fname)

fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=src, bem=bem, meg=True,
                                eeg=False, n_jobs=1)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2, depth=0.8)


def epochs_to_labels_sk(epochs, labels, inv, lambda2 = 1.0 / (1.0 ** 2), method = 'MNE'):
    n_epochs, n_chs, n_time = epochs._data.shape
    labels_data = np.zeros((len(labels), n_time, n_epochs))
    for index, label in enumerate(labels):
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, label, pick_ori="normal")
        data = np.transpose(np.array([stc.data for stc in stcs]), (1, 2, 0))
        n_verts, n_time, n_epochs = data.shape
        data = data.reshape(n_verts, n_time * n_epochs)
        U, S, V = np.linalg.svd(data, full_matrices=False)
        flip = np.array([np.sign(np.corrcoef(V[0,:],dat)[0, 1]) for dat in data])
        data = flip[:, np.newaxis] * data
        data = np.median(data.reshape(n_verts, n_time, n_epochs), axis=0)
        labels_data[index] = data
        print(float(index) / len(labels) * 100)
    return labels_data



def epochs_to_labels_mne(epochs, labels, inv, lambda2 = 1.0 / (1.0 ** 2), method = 'MNE', mode='pca_flip'):
    src = inv['src']
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, return_generator=True)
    labels_data = mne.extract_label_time_course(stcs, labels, src, mode=mode)
    labels_data = np.array(labels_data)
    labels_data = np.transpose(labels_data, (1, 2, 0))
    return labels_data


def compute_psd(labels_data, sfreq=300):
    n_labels, n_time, n_epochs = labels_data.shape
    labels_psd = np.zeros((n_labels, n_epochs, 1201))
    for ind1 in np.arange(n_labels):
        for ind2 in np.arange(n_epochs):
            labels_psd[ind1,ind2,:] = stft(labels_data[ind1,:,ind2], sfreq)[1]
            print(ind1)
    freq = stft(labels_data[ind1,:,ind2], sfreq)[0]
    return labels_psd, freq

labels_data_pf = epochs_to_labels_mne(epochs, labels, inv, mode='pca_flip')
labels_data_pfm = epochs_to_labels_mne(epochs, labels, inv, mode='pca_flip_mean')
labels_data_pft = epochs_to_labels_mne(epochs, labels, inv, mode='pca_flip_truncated')
labels_data_mf = epochs_to_labels_mne(epochs, labels, inv, mode='mean_flip')
labels_data_sk = epochs_to_labels_sk(epochs, labels, inv)

labels_psd_pf, freq = compute_psd(labels_data_pf)
labels_psd_pfm = compute_psd(labels_data_pfm)[0]
labels_psd_pft = compute_psd(labels_data_pft)[0]
labels_psd_mf = compute_psd(labels_data_mf)[0]
labels_psd_sk = compute_psd(labels_data_sk)[0]

plt.figure();plt.plot(freq[18:], np.median(labels_psd_pf,1).T[18:]);plt.title('pca flip');plt.xlim(2.5, 90)
plt.figure();plt.plot(freq[18:], np.median(labels_psd_pfm,1).T[18:]);plt.title('pca flip mean');plt.xlim(2.5, 90)
plt.figure();plt.plot(freq[18:], np.median(labels_psd_pft,1).T[18:]);plt.title('pca flip truncated');plt.xlim(2.5, 90)
plt.figure();plt.plot(freq[18:], np.median(labels_psd_mf,1).T[18:]);plt.title('Matti flip');plt.xlim(2.5, 90)
plt.figure();plt.plot(freq[18:], np.median(labels_psd_sk,1).T[18:]);plt.title('Sheraz flip');plt.xlim(2.5, 90)



