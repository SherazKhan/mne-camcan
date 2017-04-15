import os.path as op
import math

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
for hemi in ('lh', 'rh'):
    brain = surfer.Brain(
        'fsaverage', surf='inflated', cortex='bone', hemi=hemi,
        subjects_dir=subjects_dir)
    brain.add_annotation('HCPMMP1')
    brain.save_montage('hcp_parc_montage_%s.jpg' % hemi,
                       order=('lat', 'med', 'dor', 'ven'))
    brain.close()

for hemi in ('lh', 'rh'):
    brain = surfer.Brain(
        'fsaverage', surf='inflated', cortex='bone', hemi=hemi,
        subjects_dir=subjects_dir)
    brain.add_annotation('HCPMMP1_auto_split')
    brain.save_montage('hcp_parc_split_montage_%s.jpg' % hemi,
                       order=('lat', 'med', 'dor', 'ven'))
    brain.close()
