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