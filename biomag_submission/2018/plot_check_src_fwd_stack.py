import os
import os.path as op
import mne

data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')


subjects_dir = op.join(data_path, 'recons')
meg_dir = op.join(data_path, 'meg_dir')
out_dir = op.join(data_path, 'figures_qc') 
subject = 'CC110033'

surface = 'white'
spacing = 'oct6'
fwd_name = op.join(meg_dir, subject, '{}-{}-fwd.fif'.format(
                   spacing, surface))
fwd = mne.read_forward_solution(fname=fwd_name)

mag_map = mne.sensitivity_map(fwd, ch_type='mag')

brain = mag_map.plot(
    subjects_dir=subjects_dir, subject=subject,
    smoothing_steps=2,
    clim=dict(kind='percent', lims=[1, 50, 99]))

