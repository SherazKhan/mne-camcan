import os.path as op
from nipype.interfaces.freesurfer import ReconAll
from mne.parallel import parallel_func
import glob

subjects_dir = '/cluster/fusion/Sheraz/camcan/freesurfer'
camcan_path = '/cluster/transcend/MEG'


N_JOBS = 240
t1_files = op.join(camcan_path + '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-' + '*',
             'anat', 'sub-' + '*' + '_T1w.nii.gz')


t1_files = glob.glob(t1_files)



def process_subject_anatomy(t1):
    reconall = ReconAll()
    reconall.inputs.subject_id = t1.split('/')[-3]
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = subjects_dir
    reconall.inputs.T1_files = t1
    reconall.run()


parallel, run_func, _ = parallel_func(process_subject_anatomy, n_jobs=N_JOBS)
parallel(run_func(t1) for t1 in t1_files)



import mne
from mne.surface import decimate_surface  # noqa
from scipy.io import loadmat
import numpy as np
from mayavi import mlab  # noqa
from scipy import linalg


print(__doc__)

path = mne.datasets.sample.data_path()
surf = mne.read_bem_surfaces('/autofs/cluster/transcend/sheraz/Dropbox/mne-camcan-data/recons/CC110033/bem/CC110033-head.fif')[0]

points, triangles = surf['rr'], surf['tris']


head_col = (0.95, 0.83, 0.83)  # light pink


Source space : MRI voxel -> MRI (surface RAS)
     0.010000  0.000000  0.000000    -110.00 mm
     0.000000  0.010000  0.000000    -120.00 mm
     0.000000  0.000000  0.010000    -100.00 mm
     0.000000  0.000000  0.000000       1.00
MRI volume : MRI voxel -> MRI (surface RAS)
    -0.001000  0.000000  0.000000     128.00 mm
     0.000000  0.000000  0.001000    -128.00 mm
     0.000000 -0.001000  0.000000     128.00 mm
     0.000000  0.000000  0.000000       1.00
MRI volume : MRI (surface RAS) -> RAS (non-zero origin)
     1.000000 -0.000000 -0.000000       6.02 mm
     0.000000  1.000000 -0.000000      30.04 mm
     0.000000  0.000000  1.000000     -25.84 mm
     0.000000  0.000000  0.000000       1.00




p, t = points, triangles
mlab.triangular_mesh(p[:, 0], p[:, 1], p[:, 2], t, color=head_col, opacity=0.3)


dat = loadmat('/autofs/cluster/fusion/Sheraz/data/camcan/camcan47/cc700/meg/pipeline/release004/fiducials/CC110033/fiducials_CC110033.mat')


trans = np.array([[0.930388,  -0.00500518,  -0.0436328,  49.6375],
                    [-0.0750492,  -0.161368,  -0.860765,  228.465],
                  [-0.00662416,  0.976593,  -0.165307,  14.6245],
                  [0,  0,  0,  1]])

trans1 = np.array([[-1,  0,  0,  129.20779],
                    [0,  0,  1,  -93.113],
                  [0, -1,  0,  118.11857],
                  [0,  0,  0,  1]])


trans2 = np.array([[-1,  0,  0, 128],
                    [0,  0,  1,  -128],
                  [0, -1,  0,  128],
                  [0,  0,  0,  1]])

mri_info /autofs/cluster/transcend/sheraz/Dropbox/mne-camcan-data/recons/CC110033/mri/T1.mgz --vox2ras
register mni152reg --s CC110033 --1





fud = np.hstack((dat['M'],np.ones((3,1))))

np.dot(trans1, np.dot(trans,fud[2]))

mlab.points3d(p1[0],p1[1],p1[2], color=(1, 1, 0), scale_factor=0.01)





def translation(x=0, y=0, z=0):
    """Create an array with a translation matrix.
    Parameters
    ----------
    x, y, z : scalar
        Translation parameters.
    Returns
    -------
    m : array, shape = (4, 4)
        The translation matrix.
    """
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m


def get_ras_to_neuromag_trans(nasion, lpa, rpa):
    """Construct a transformation matrix to the MNE head coordinate system.
    Construct a transformation matrix from an arbitrary RAS coordinate system
    to the MNE head coordinate system, in which the x axis passes through the
    two preauricular points, and the y axis passes through the nasion and is
    normal to the x axis. (see mne manual, pg. 97)
    Parameters
    ----------
    nasion : array_like, shape (3,)
        Nasion point coordinate.
    lpa : array_like, shape (3,)
        Left peri-auricular point coordinate.
    rpa : array_like, shape (3,)
        Right peri-auricular point coordinate.
    Returns
    -------
    trans : numpy.array, shape = (4, 4)
        Transformation matrix to MNE head space.
    """
    # check input args
    nasion = np.asarray(nasion)
    lpa = np.asarray(lpa)
    rpa = np.asarray(rpa)
    for pt in (nasion, lpa, rpa):
        if pt.ndim != 1 or len(pt) != 3:
            raise ValueError("Points have to be provided as one dimensional "
                             "arrays of length 3.")

    right = rpa - lpa
    right_unit = right / linalg.norm(right)

    origin = lpa + np.dot(nasion - lpa, right_unit) * right_unit

    anterior = nasion - origin
    anterior_unit = anterior / linalg.norm(anterior)

    superior_unit = np.cross(right_unit, anterior_unit)

    x, y, z = -origin
    origin_trans = translation(x, y, z)

    trans_l = np.vstack((right_unit, anterior_unit, superior_unit, [0, 0, 0]))
    trans_r = np.reshape([0, 0, 0, 1], (4, 1))
    rot_trans = np.hstack((trans_l, trans_r))

    trans = np.dot(rot_trans, origin_trans)
    return trans




linalg.inv(get_ras_to_neuromag_trans([-5, 72, -39.4], [-73, -4.2, -39.6], [72.4, -14.6, -46.9]))







































































