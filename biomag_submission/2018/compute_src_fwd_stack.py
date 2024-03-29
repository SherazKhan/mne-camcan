import os
import os.path as op
import mne
from joblib import Parallel, delayed

data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')

subjects_dir = op.join(data_path, 'recons')
meg_dir = op.join(data_path, 'meg_dir')


def _run_make_anatomy(subject, subjects_dir, surface='white',
                      spacings=('oct6', 'ico5')):
    for spacing in spacings:
        src = mne.setup_source_space(
            subject=subject,
            spacing=spacing, surface=surface, subjects_dir=subjects_dir,
            add_dist=False, n_jobs=1, verbose=None)
        out_path = op.join(data_path, 'meg_dir', subject)
        if not op.exists(out_path):
            os.makedirs(out_path)
        out_name_src = op.join(out_path, '{}-{}-src.fif'.format(
            surface, spacing))
        mne.write_source_spaces(out_name_src, src, overwrite=True)

        trans = mne.read_trans(
            op.join(data_path, 'trans', subject + '-trans.fif'))

        conductivity = (0.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        bem_fname = op.join(
            subjects_dir, subject, 'bem', '%s-src.fif' % subject)
        mne.write_bem_solution(bem_fname, bem=bem)
        info = mne.io.read_info(
            op.join(meg_dir, subject,
                    'rest-sss-raw.fif'))
        fwd = mne.make_forward_solution(
            info=info, trans=trans, src=src, bem=bem, meg=True, eeg=False)
        fwd_name = op.join(meg_dir, subject, '{}-{}-{}-fwd.fif'.format(
            spacing, surface, spacing))
        mne.write_forward_solution(fname=fwd_name, fwd=fwd, overwrite=True)


subjects = ['CC110033', 'CC110037', 'CC110045']


out = Parallel(n_jobs=8)(delayed(_run_make_anatomy)(
    subject=subject, subjects_dir=subjects_dir) for subject in subjects)
