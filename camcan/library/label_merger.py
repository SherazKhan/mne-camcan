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



def compute_area(label, subject):
    command = 'mris_anatomical_stats -l ' + label + ' ' + subject + ' ' + label[-8:-6] + ' inflated >& ' + log_file
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
from __future__ import division
import mne
import numpy as np
import os.path as op
import glob
import matplotlib.pyplot as plt
plt.ion()
data_path = '/cluster/transcend/sheraz/Dropbox/mne-camcan-data/'

subjects_dir = op.join(data_path,'recons')


labels_fname  = sorted(glob.glob(op.join(data_path, 'labels', '*lh.label')))
labels = [mne.read_label(label, subject='fsaverage', color='r')
          for label in labels_fname]

labels_size = np.array([label.vertices.shape[0] for label in labels])
























































































