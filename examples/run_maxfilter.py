import mne
import os.path as op
from autoreject import Ransac  # noqa
import matplotlib.pyplot as plt
from camcan.library.config import ctc, cal
from mne.preprocessing import ICA

subjects = ['CC110033', 'CC110037', 'CC110045']

data_path = '/home/sheraz/Dropbox/mne-camcan-data'
event_id = 1
event_overlap = 4
event_length = 30
subject = subjects[0]

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


raw.filter(2, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

reject = dict(grad=4000e-13, mag=4e-12)
events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap, start=0, stop=raw_length-event_length)
epochs = mne.Epochs(raw, events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)



