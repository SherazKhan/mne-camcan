import os.path as op
import math

import glob
import numpy as np

import surfer

import mne
from mne.datasets.brainstorm import bst_resting

data_path = bst_resting.data_path()
subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)


def save_parc_montage(parc):
    for hemi in ('lh', 'rh'):
        brain = surfer.Brain(
            'fsaverage', surf='inflated', cortex='bone', hemi=hemi,
            subjects_dir=subjects_dir)
        brain.add_annotation(parc)
        brain.save_montage('%s_%s.jpg' % (parc, hemi),
                           order=('lat', 'med', 'dor', 'ven'))
        brain.close()

###############################################################################
# HCP


labels = mne.read_labels_from_annot(
    parc='HCPMMP1', subject='fsaverage', subjects_dir=subjects_dir)

label_dict = {
    lh.name[:-3]: (lh, rh) for lh, rh in zip(labels[::2], labels[1::2])}

split_map = dict()
for key, value in label_dict.items():
    split_map[key] = int(math.ceil(np.max(
        [label.vertices.shape[0] for label in value]) / 1e3))

split_labels = list()
for key, value in label_dict.items():
    n_parts = split_map[key]
    for label in value:
        new_labels = label.split(n_parts) if n_parts > 1 else [label]
        split_labels.extend(new_labels)

# trick to avoid non-unique colors which would crash
colors = np.array([ll.color for ll in split_labels])
rand_X = np.random.sample(len(colors) * 4).reshape(-1, 4)
colors += ((rand_X - 0.5) * 0.05)
for ii, color in enumerate(colors):
    split_labels[ii].color = color

mne.write_labels_to_annot(
    split_labels, subject='fsaverage', subjects_dir=subjects_dir,
    parc='HCPMMP1_auto_split', overwrite=True)

# write out things

save_parc_montage(parc='HCPMMP1')
save_parc_montage(parc='HCPMMP1_auto_split')

###############################################################################
# Sheraz Khan

camcan_data_path = '~/Dropbox/mne-camcan-data'
sheraz_labels_fnames = glob.glob(
    op.expanduser(op.join(camcan_data_path, 'labels', '*.label')))

sheraz_labels = [mne.read_label(fname, subject='fsaverage')
                 for fname in sheraz_labels_fnames]

mne.write_labels_to_annot(
    sheraz_labels, subject='fsaverage', subjects_dir=subjects_dir,
    parc='aparc_sk', overwrite=True)
sheraz_labels

save_parc_montage(parc='aparc_sk')
