# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 03:25:12 2014

@author: sheraz
"""

import glob
from surfer import Brain
import matplotlib.pyplot as plt
import numpy as np

subject_id = "fsaverage"
hemi = "lh"
surf = "inflated_pre"
brain = Brain(subject_id, hemi, surf)
labels = glob.glob("/home/sheraz/Dropbox/mne-camcan-data/label68/lh*.label")
labels.sort()
cmap = plt.cm.get_cmap('Set3', 34)
cmaplist = [cmap(i) for i in range(cmap.N)]

for i in range(34):
    brain.add_label(labels[i], (0.7 * np.random.rand(1, 1) + 0.3, np.random.rand(1, 1), np.random.rand(1, 1)),
                    alpha=0.85)

brain.show_view("lateral")
brain.save_image("lr1.tiff")

brain.show_view("medial")
brain.save_image("lm1.tiff")


n_labels = len(labels)
label_names = ['L%03d' % ii for ii in range(n_labels)]

info = mne.create_info(
    ch_names=label_names, sfreq=300, ch_types=['misc'] * n_labels)

info['highpass'] = raw.info['highpass']
info['lowpass'] = raw.info['lowpass']
for ch in info['chs']:
    ch['unit'] = FIFF.FIFF_UNIT_AM  # put ampere meter
    ch['unit_mul'] = FIFF.FIFF_UNIT_NONE  # no unit multiplication

label_raw = mne.io.RawArray(label_raw, info)
label_raw.annotations = raw.annotations

















