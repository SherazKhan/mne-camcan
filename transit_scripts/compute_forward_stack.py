import os.path as op

import mne
from mne.datasets.brainstorm import bst_resting

data_path = bst_resting.data_path()
raw_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_spontaneous_20111102_02_AUX_raw.fif')

raw_noise_fname = op.join(
    data_path,
    'MEG/bst_resting/subj002_noise_20111104_02_raw.fif')

subject = 'bst_resting'
subjects_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/subjects')
meg_dir = op.expanduser(
    '~/mne_data/MNE-brainstorm-data/bst_resting/MEG')

spacing = 'ico5'
proj_fname = op.join(meg_dir, subject, 'bst_resting_ecg-eog-proj.fif')

raw = mne.io.read_raw_fif(raw_fname)
raw.rename_channels(
    dict(zip(raw.ch_names, mne.utils._clean_names(raw.ch_names))))
raw.load_data()

picks = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=True)
raw.filter(1, 200, picks=picks)
projs = mne.read_proj(proj_fname)
raw.add_proj(projs)

bem_fname = op.join(subjects_dir, subject, 'bem', '%s-bem.fif' % subject)
src_fname = op.join(
    meg_dir, subject, '%s-%s-src.fif' % (subject, spacing))

src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)

trans_fname = op.join(meg_dir, subject, "bs_resting_01-trans.fif")

for comp in raw.info['comps']:
    for key in ('row_names', 'col_names'):
        comp['data'][key] = mne.utils._clean_names(comp['data'][key])

fwd = mne.make_forward_solution(raw.info, trans_fname, src=src, bem=bem,
                                eeg=False)

mne.write_forward_solution(
    op.join(meg_dir, subject, '%s_%s-fwd.fif' % (subject, spacing)), fwd,
    overwrite=True)

raw_noise = mne.io.read_raw_fif(raw_noise_fname)
raw_noise.rename_channels(
    dict(zip(raw_noise.ch_names,
             mne.utils._clean_names(raw_noise.ch_names))))

raw_noise.add_proj(projs)
raw_noise.load_data()
picks = mne.pick_types(raw_noise.info, meg=True, eeg=False, ref_meg=True)
raw_noise.filter(1, 200, picks)
noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')
mne.write_cov(op.join(meg_dir, subject, '%s-%s-cov.fif' % (subject, spacing)),
              noise_cov)

assert noise_cov['names'] == fwd['info']['ch_names']
