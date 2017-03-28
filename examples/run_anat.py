import mne
import subprocess
import os.path as op

recon_dir = '/autofs/cluster/fusion/Sheraz/camcan/recons'
camcan_path = '/cluster/transcend/MEG'
subj = 'CC110037'

COMMAND = ['setenv', 'SUBJECTS_DIR', recon_dir, 'recon-all', '-all', '-s', subj, '-i',
           op.joins(camcan_path, '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-'+subj,
                    'anat','sub'+subj+'_T1w.nii.gz')]


p = subprocess.Popen(COMMAND,
                     shell=False,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
(output, err) = p.communicate()
