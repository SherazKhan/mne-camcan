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
labels = glob.glob("/cluster/transcend/sheraz/fix/labels/*lh.label")
labels.sort()
cmap = plt.cm.get_cmap('prism', 225)
cmaplist = [cmap(i) for i in range(cmap.N)]

for i in range(225):
    brain.add_label(labels[i], (0.7 * np.random.rand(1, 1) + 0.3, np.random.rand(1, 1), np.random.rand(1, 1)),
                    alpha=0.7)

brain.show_view("lateral")
brain.save_image("lr1.tiff")

brain.show_view("medial")
brain.save_image("lm1.tiff")