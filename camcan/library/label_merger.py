from __future__ import division
import os
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
from surfer import Brain
from pyimpress.utils import ParallelExecutor
from joblib import delayed
plt.ion()

subject = 'AC077'

labels_path = '/autofs/cluster/transcend/sheraz/rs/452/AC077/*.label'
log_file = '/autofs/cluster/transcend/sheraz/rs/452/AC077/log.log'

labels = glob.glob(labels_path)
subjects_dir = '/cluster/transcend/MRI/WMA/recons'


def compute_area(label, subject):
    command = 'mris_anatomical_stats -l ' + label + ' ' + subject + ' ' + label[-8:-6] + ' white >& ' + log_file
    status = os.system(command)

    if not bool(status):
        datafile = file(log_file)
        for line in datafile:
            if 'total surface area' in line:
                datafile.close()
                return int(line.split(' ')[-2])





n_jobs = 28
aprun = ParallelExecutor(n_jobs=n_jobs)
labels_area = aprun(total=len(labels))(delayed(compute_area)(label, subject) for label in labels)




labelss = [mne.read_label(label, subject=subject, color='r')
          for label in labels]



x = np.array(area[:], dtype=np.int32)

l1, l2 = mne.label.split_label(mne.read_label('/autofs/cluster/transcend/sheraz/rs/452/AC077/AC077-supramarginal_7-lh.label',
                                              subject='AC077'), parts=2, subject='AC077')
lab = mne.read_label('/autofs/cluster/transcend/sheraz/rs/452/AC077/AC077-supramarginal_7-lh.label',
                                              subject='AC077')
l1.save('/autofs/cluster/transcend/sheraz/rs/452/AC077/AC077-supramarginal_7a-lh.label')
l2.save('/autofs/cluster/transcend/sheraz/rs/452/AC077/AC077-supramarginal_7b-lh.label')

hemi = "lh"
surf = "inflated"
brain = Brain(subject, hemi, surf)

brain.add_label(l2, (0,0,1))

brain.add_label(l2, (0,0,1))

brain.add_label(lab, (0,1,0),borders=True)


##
from __future__ import division
import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
plt.ion()
data_path = '/cluster/transcend/sheraz/Dropbox/mne-camcan-data/'

subjects_dir = op.join(data_path,'recons')


labels = mne.read_labels_from_annot('CC110033', parc='aparc', subjects_dir=subjects_dir)

labels_size = np.array([label.vertices.shape[0] for label in labels])

plt.hist(labels_size, 30)
labels_size.min()


new_labels = []



##

import mne
import os
import numpy as np
import os.path as op
from mne.datasets import sample
from nose.tools import assert_equal


data_path = sample.data_path()

meg_path = data_path + '/MEG/sample'

subject = 'sample'
subjects_dir = data_path + '/subjects'

label_fname  = meg_path + '/labels/Aud-lh.label'
label = mne.read_label(label_fname, subject=subject)



def get_vertices_faces(vertices_ind, subjects_dir, subject, hemi, surf):

    surf = mne.surface.read_surface(op.join(subjects_dir, subject, 'surf', hemi + '.' + surf))
    ind = np.in1d(surf[1][:,0], vertices_ind)*\
        np.in1d(surf[1][:,1], vertices_ind)*\
        np.in1d(surf[1][:,2], vertices_ind)

    vertices = surf[0][vertices_ind]
    faces = surf[1][ind]
    faces = np.array([np.where(vertices_ind == face)[0]
                      for face in faces.ravel()]).reshape(faces.shape)

    return vertices, faces


def triangle_area(vertices, faces):

    r12 = vertices[faces[:,0],:]
    r13 = vertices[faces[:,2],:] - r12
    r12 = vertices[faces[:,1],:] - r12
    return np.sum(np.sqrt(np.sum(np.cross(r12, r13)**2,axis=1))/2.)



def compute_area(label_fname, subject):
    log_file = op.join(subjects_dir, subject, 'label', 'temp.log')
    command = 'mris_anatomical_stats -l ' + label_fname + ' ' + subject + ' ' + label_fname[-8:-6] + ' white >& ' + log_file
    status = os.system(command)

    if not bool(status):
        datafile = file(log_file)
        for line in datafile:
            if 'total surface area' in line:
                datafile.close()
                return int(line.split(' ')[-2])
        os.remove(log_file)


vertices, faces = get_vertices_faces(label.vertices, subjects_dir, label.subject, label.hemi, 'white')

label_area_mne = triangle_area(vertices, faces)

label_area_freesurfer = compute_area(label_fname, subject)


def test_triangle_area():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    assert_equal(triangle_area(vertices, faces), 1)

































































