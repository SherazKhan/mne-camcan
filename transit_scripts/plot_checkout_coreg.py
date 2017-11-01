import os.path as op

import mne

import numpy as np
from scipy.io import loadmat

subject = 'CC110033'

mat_name = op.expanduser(
    '~/Dropbox/mne-camcan-data/fiducials/{sub}/fiducials_{sub}.mat'.format(
        sub=subject
    ))

mni152_trans_name = op.expanduser(
    '~/Dropbox/mne-camcan-data/recons/{sub}/mri/'
    'transforms/reg.mni152.1mm.dat.fsl.mat'.format(
        sub=subject))


mat = loadmat(mat_name)

ident = [mne.io.constants.FIFF.FIFFV_POINT_LPA,
         mne.io.constants.FIFF.FIFFV_POINT_RPA,
         mne.io.constants.FIFF.FIFFV_POINT_NASION]

mne.io.constants.FIFF.FIFFV_COORD_MRI

fids = np.hstack((mat['Mmm'] / 1000, np.ones((3, 1))))[:, :3]

trans152 = np.loadtxt(mni152_trans_name)

fiducials = [{'coord_frame': 5,
              'ident': ident[ii], 'kind': 1, 'r': rr}
             for ii, rr in enumerate(fids)]

info = mne.io.read_info(
    '/Users/dengeman/Dropbox/mne-camcan-data/'
    'rest/sub-%s/meg/rest_raw.fif' % subject)

trans = mne.coreg.coregister_fiducials(info, fiducials)
mne.viz.plot_alignment(
    info, trans=trans, subject=subject,
    subjects_dir='/Users/dengeman/Dropbox/mne-camcan-data/recons')
