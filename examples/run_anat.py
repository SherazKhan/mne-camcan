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







np.dot(trans1, np.dot(trans,fud[2]))

mlab.points3d(p1[0],p1[1],p1[2], color=(1, 1, 0), scale_factor=0.01)


from mne.transforms import get_ras_to_neuromag_trans
from scipy.io import loadmat
import os


dat = loadmat('/autofs/cluster/fusion/Sheraz/data/camcan/camcan47/cc700/meg/pipeline/release004/fiducials/CC110033/fiducials_CC110033.mat')
fud = np.hstack((dat['M'],np.ones((3,1))))
trans = np.loadtxt('/autofs/cluster/transcend/sheraz/Dropbox/mne-camcan-data/recons/CC110033/mri/transforms/reg.mni152.1mm.dat.fsl.mat')

os.system('mni152reg --s CC110033 --1')

trans2 = np.array([[-1,  0,  0, 128],
                    [0,  0,  1,  -128],
                  [0, -1,  0,  128],
                  [0,  0,  0,  1]])



trans = linalg.inv(get_ras_to_neuromag_trans(np.dot(trans2, np.dot(trans,fud[2]))[:3], np.dot(trans2, np.dot(trans,fud[1]))[:3], np.dot(trans2, np.dot(trans,fud[0]))[:3]))







































































