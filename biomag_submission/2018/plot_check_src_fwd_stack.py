import os.path as op
import mne

inp_data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')
out_data_path = op.expanduser(
    '~/github/mne-camcan/biomag_submission/2018')

subjects_dir = op.join(inp_data_path, 'recons')
meg_dir = op.join(inp_data_path, 'meg_dir')
out_dir = op.join(out_data_path, 'figures_qc')
surface = 'white'
spacing = 'oct6'

subjects = ['CC110033', 'CC110037', 'CC110045']
for subject in subjects:

    fwd_name = op.join(meg_dir, subject, '{}-{}-fwd.fif'.format(
                       spacing, surface))
    fwd = mne.read_forward_solution(fname=fwd_name)

    mag_map = mne.sensitivity_map(fwd, ch_type='mag')

    brain = mag_map.plot(
        subjects_dir=subjects_dir, subject=subject,
        smoothing_steps=2,
        clim=dict(kind='percent', lims=[1, 50, 99]))

    brain.save_montage(
        op.join(out_dir, '{}-fwd-check.png'.format(subject)))
