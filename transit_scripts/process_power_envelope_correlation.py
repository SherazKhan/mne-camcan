# cd transit_scripts/
import os.path as op
import mne
import bct
import numpy as np

corr = mne.externals.h5io.read_hdf5('beta_band_brain_power_envelopes_beta_wide.h5')['C']
corr = np.median(corr, axis=0)
corr = np.int32(bct.utils.threshold_proportional(corr, .15) > 0)
deg = bct.degrees_und(corr)

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')

labels = mne.read_labels_from_annot(
    parc='aparc_sk', subject='fsaverage', subjects_dir=subjects_dir)

labels = [ll for ll in labels if 'unknown' not in ll.name]


def labels2stc(labels, labels_data, stc):
    stc_new = stc.copy()
    stc_new.data.fill(0)
    for index, label in enumerate(labels):
        if labels_data.ndim == 1:
            temp = stc.in_label(label)
            temp.data.fill(labels_data[index])
            stc_new += temp.expand(stc.vertices)
        else:
            lab = mne.read_label(label)
            ver = np.intersect1d(lab.vertices, stc.vertices)
            if '-rh' in label:
                ver += len(stc.vertices[0])
            stc_data = np.tile(labels_data[index][:, np.newaxis], len(ver)).T
            stc_new.data[ver, :] = stc_data
    return stc_new


def get_stc(labels, data, tmin=0, tstep=1):
    stc_vertices = [np.uint32(np.arange(10242)),
                    np.uint32(np.arange(10242))]

    if data.ndim == 1:
        stc_data = np.ones((20484, 1), dtype=np.float32)
    else:
        stc_data = np.ones((20484, data.shape[1]), dtype=np.float32)

    stc = mne.SourceEstimate(
        stc_data, vertices=stc_vertices, tmin=tmin, tstep=tstep,
        subject='fsaverage')
    stc_new = labels2stc(labels, data, stc)
    return stc_new


stc_new = get_stc(labels, deg)

brain = stc_new.plot(
    subject='fsaverage', hemi='split', views=['med', 'lat'], colormap='gnuplot',
    surface='inflated10', subjects_dir=subjects_dir, time_label=None)
brain.scale_data_colormap(80, 120, 180, True)
brain.save_image('ctf_brainstorm_wide_beta_.jpg')
