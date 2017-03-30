import mne
import os.path as op
from autoreject import Ransac  # noqa
import matplotlib.pyplot as plt
from camcan.library.config import ctc, cal
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.label import Label, _blend_colors, label_sign_flip
import glob

subjects = ['CC110033', 'CC110037', 'CC110045']
subject = subjects[0]

data_path = '/home/sheraz/Dropbox/mne-camcan-data'
subjects_dir = op.join(data_path,'recons')
subject_dir = op.join(subjects_dir,subject)
bem_dir = op.join(subject_dir,'bem')
trans_file = op.join(data_path, 'trans',subject + '-trans.fif')
labels  = glob.glob(op.join(data_path, 'labels', '*.label'))
labels = [mne.read_label(label, subject='fsaverageSK', color='r')
          for label in labels]
for index, label in enumerate(labels):
    label.values.fill(1.0)
    labels[index] = label

labels = [label.morph('fsaverageSK', subject, subjects_dir=subjects_dir) for label in labels]

event_id = 1
event_overlap = 4
event_length = 30
spacing='ico5'


#def process_maxfilter(subject):
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

raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

raw.filter(2, None, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

reject = dict(grad=4000e-13, mag=4e-12)
events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap, start=0, stop=raw_length-event_length)
epochs = mne.Epochs(raw, events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
epochs.resample(150.)

bem_fname = op.join(bem_dir, '%s-src.fif' % subject)
src_fname = op.join(bem_dir, '%s-src.fif' % spacing)

bem = mne.read_bem_solution(bem_fname)
src = mne.read_source_spaces(src_fname)

fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=src, bem=bem,fname=None, meg=True, eeg=False, n_jobs=2)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2, depth=0.8)

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "MNE"
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method,
                            pick_ori="normal", return_generator=False)



