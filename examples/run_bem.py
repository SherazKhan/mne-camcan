import mne
from mne.parallel import parallel_func
import os.path as op

subjects_dir = '/cluster/fusion/Sheraz/camcan/recons'


subjects = ['CC110033', 'CC110037', 'CC110045']
N_JOBS = 3

def process_subject_bem(subject, spacing='ico5'):
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir)
    conductivity = (0.3,)
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False, overwrite=True)
    bem_fname = op.join(subjects_dir,subject,'bem', '%s-src.fif' % subject)
    src_fname = op.join(subjects_dir, subject, 'bem', '%s-src.fif' % spacing)
    mne.write_bem_solution(bem_fname, bem=bem)
    mne.write_source_spaces(src_fname, src=src)


parallel, run_func, _ = parallel_func(process_subject_bem, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in subjects)
