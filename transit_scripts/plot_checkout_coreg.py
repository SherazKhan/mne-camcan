import os.path as op

import mne

import numpy as np
from scipy.io import loadmat

subject = 'CC110045'
subjects_dir = op.expanduser(
    '~/Dropbox/mne-camcan-data/recons')

mat_name = op.expanduser(
    '~/Dropbox/mne-camcan-data/fiducials/{sub}/fiducials_{sub}.mat'.format(
        sub=subject
    ))

""" From Darren's e-mail.

I have added a new folder to Sheraz’s account containing the fiducial markers.
Here is an example, where L.M is the voxel positions for the resliced image
(using MNI152_T1_1mm.nii, you don’t have that image so you would need to
recreate it), and Mmm is the fiducial positions in mm after a rigid body
coregistration to MNI space (the files you have should be in this space).
This is the one you should use. The reason for this was that when doing the
manual coreg, it was much easier when the images were resliced in MNI space.
If you want native space Voxel indices then you would need to apply the inverse
of the transformation matrix in the NII images that you have. Let me know if
any of that is not clear.

L = load(‘fiducials/CC110045/FeducialLocs.mat')

>> L.Mmm

ans =

   -75   -25   -37 < Left (X Y Z)
    75   -25   -41 < Right
     0    78   -37 < Nasion
"""

mat = loadmat(mat_name)

ident = [mne.io.constants.FIFF.FIFFV_POINT_LPA,
         mne.io.constants.FIFF.FIFFV_POINT_RPA,
         mne.io.constants.FIFF.FIFFV_POINT_NASION]

#
# Xfm_mni2ras = np.array(
#     [[0.9975, -0.0073, 0.0176, -0.0429],
#      [0.0146,  1.0009, -0.0024, 1.5496],
#      [-0.0130, -0.0093, 0.9971, 1.1840],
#      [0, 0, 0, 1.]]
# )
#

# Xfm_mni2ras = np.array(
#     [[0.88, 0, 0, -0.8],
#      [0, 0.97, 0, -3.32],
#      [0, 0.05, 0.88, -0.44],
#      [0, 0, 0, 1]]
# )
#
# trans_mni2ras = mne.transforms.Transform(
#     to=mne.io.constants.FIFF.FIFFV_COORD_MRI,
#     fro=mne.io.constants.FIFF.FIFFV_MNE_COORD_MNI_TAL,
#     trans=Xfm_mni2ras)


def _get_mri_header(fname):
    """Get MRI header using nibabel."""
    import nibabel as nib
    img = nib.load(fname)
    try:
        return img.header
    except AttributeError:  # old nibabel
        return img.get_header()


def _read_talxfm(subject, subjects_dir, mode=None, verbose=None):
    """Read MNI transform from FreeSurfer talairach.xfm file.

    Adapted from freesurfer m-files. Altered to deal with Norig
    and Torig correctly.
    """
    from mne.transforms import Transform, combine_transforms, invert_transform
    from mne.io.constants import FIFF
    from mne.utils import has_nibabel,  run_subprocess, logger
    if mode is not None and mode not in ['nibabel', 'freesurfer']:
        raise ValueError('mode must be "nibabel" or "freesurfer"')
    fname = op.join(subjects_dir, subject, 'mri', 'transforms',
                    'talairach.xfm')
    # read the RAS to MNI transform from talairach.xfm
    with open(fname, 'r') as fid:
        logger.debug('Reading FreeSurfer talairach.xfm file:\n%s' % fname)

        # read lines until we get the string 'Linear_Transform', which precedes
        # the data transformation matrix
        got_it = False
        comp = 'Linear_Transform'
        for line in fid:
            if line[:len(comp)] == comp:
                # we have the right line, so don't read any more
                got_it = True
                break

        if got_it:
            xfm = list()
            # read the transformation matrix (3x4)
            for ii, line in enumerate(fid):
                digs = [float(s) for s in line.strip('\n;').split()]
                xfm.append(digs)
                if ii == 2:
                    break
            xfm.append([0., 0., 0., 1.])
            xfm = np.array(xfm, dtype=float)
        else:
            raise ValueError('failed to find \'Linear_Transform\' string in '
                             'xfm file:\n%s' % fname)

    # Setup the RAS to MNI transform
    ras_mni_t = {'from': FIFF.FIFFV_MNE_COORD_RAS,
                 'to': FIFF.FIFFV_MNE_COORD_MNI_TAL, 'trans': xfm}

    # now get Norig and Torig
    # (i.e. vox_ras_t and vox_mri_t, respectively)
    path = op.join(subjects_dir, subject, 'mri', 'orig.mgz')
    if not op.isfile(path):
        path = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if not op.isfile(path):
        raise IOError('mri not found: %s' % path)

    if has_nibabel():
        use_nibabel = True
    else:
        use_nibabel = False
        if mode == 'nibabel':
            raise ImportError('Tried to import nibabel but failed, try using '
                              'mode=None or mode=Freesurfer')

    # note that if mode == None, then we default to using nibabel
    if use_nibabel is True and mode == 'freesurfer':
        use_nibabel = False
    if use_nibabel:
        hdr = _get_mri_header(path)
        # read the MRI_VOXEL to RAS transform
        n_orig = hdr.get_vox2ras()
        # read the MRI_VOXEL to MRI transform
        ds = np.array(hdr.get_zooms())
        ns = (np.array(hdr.get_data_shape()[:3]) * ds) / 2.0
        t_orig = np.array([[-ds[0], 0, 0, ns[0]],
                           [0, 0, ds[2], -ns[2]],
                           [0, -ds[1], 0, ns[1]],
                           [0, 0, 0, 1]], dtype=float)
        nt_orig = [n_orig, t_orig]
    else:
        nt_orig = list()
        for conv in ['--vox2ras', '--vox2ras-tkr']:
            stdout, stderr = run_subprocess(['mri_info', conv, path])
            stdout = np.fromstring(stdout, sep=' ').astype(float)
            if not stdout.size == 16:
                raise ValueError('Could not parse Freesurfer mri_info output')
            nt_orig.append(stdout.reshape(4, 4))
    # extract the MRI_VOXEL to RAS transform
    n_orig = nt_orig[0]
    vox_ras_t = {'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                 'to': FIFF.FIFFV_MNE_COORD_RAS,
                 'trans': n_orig}

    # # extract the MRI_VOXEL to MRI transform
    t_orig = nt_orig[1]
    vox_mri_t = Transform('mri_voxel', 'mri', t_orig)
    #
    # # invert MRI_VOXEL to MRI to get the MRI to MRI_VOXEL transform
    mri_vox_t = invert_transform(vox_mri_t)
    #
    # # construct an MRI to RAS transform
    mri_ras_t = combine_transforms(mri_vox_t, vox_ras_t, 'mri', 'ras')
    #
    # # construct the MRI to MNI transform
    # mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t, 'mri', 'mni_tal')
    # XXX figure out what is needed here.

    mni_ras_t = invert_transform(ras_mni_t)
    ras_mri_t = invert_transform(mri_ras_t)
    mni_mri_t = combine_transforms(
        mni_ras_t, ras_mri_t, FIFF.FIFFV_MNE_COORD_MNI_TAL, 'mri')
    return mni_mri_t


mne.io.constants.FIFF.FIFFV_COORD_MRI
mni_mri_t = _read_talxfm(subject, subjects_dir=subjects_dir)


# mri_mri_t = mne.transforms.invert_transform(mri_mri_t)
# mne.transforms.combine_transforms(
#     trans, trans_mni2ras, 'head', mne.io.constants.FIFF.FIFFV_COORD_MRI)

fids = mne.transforms.apply_trans(mni_mri_t, mat['Mmm'] / 1000.)

fiducials = [{'coord_frame': 5,
              'ident': ident[ii], 'kind': 1, 'r': rr}
             for ii, rr in enumerate(fids)]

info = mne.io.read_info(
    '/Users/dengeman/Dropbox/mne-camcan-data/'
    'rest/sub-%s/meg/rest_raw.fif' % subject)

trans = mne.coreg.coregister_fiducials(info, fiducials)

mne.write_trans(
    '/Users/dengeman/Dropbox/mne-camcan-data/rest/'
    'sub-%s/meg/fid-trans.fif' % subject, trans)


mne.viz.plot_trans(info, trans=trans, subject=subject, subjects_dir=subjects_dir)
