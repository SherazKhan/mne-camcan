import os.path as op
import mne

subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')


conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='bst_resting', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
