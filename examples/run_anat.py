import os.path as op
from nipype.interfaces.freesurfer import ReconAll
from mne.parallel import parallel_func


subjects_dir = '/cluster/fusion/Sheraz/camcan/recons'
camcan_path = '/cluster/transcend/MEG'

subjects = ['CC110033', 'CC110037', 'CC110045']
N_JOBS = 3


def process_subject_anatomy(subject):

    t1_fname = op.join(camcan_path + '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-' + subject,
             'anat', 'sub-' + subject + '_T1w.nii.gz')

    t2_fname = op.join(camcan_path + '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-' + subject,
             'anat', 'sub-' + subject + '_T2w.nii.gz')

    reconall = ReconAll()
    reconall.inputs.subject_id = subject
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = subjects_dir
    reconall.inputs.T1_files = t1_fname
    reconall.inputs.T1_files = t2_fname
    reconall.run()

parallel, run_func, _ = parallel_func(process_subject_anatomy, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in subjects)



