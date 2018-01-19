import os.path as op
import matplotlib.pyplot as plt
import mne


data_path = op.expanduser(
    '~/study_data/sk_de_labelsci2018/mne-camcan-data')

meg_dir = op.join(data_path, 'meg_dir')
kinds = ['rest', 'task', 'emptyroom']
subjects = ['CC110033', 'CC110037', 'CC110045']


def _run_maxfilter(raw, coord_frame='head'):
    cal = op.join('../../camcan/library/sss_cal.dat')
    ctc = op.join('../../camcan/library/ct_sparse.fif')
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=cal,
        cross_talk=ctc,
        st_duration=10.,
        st_correlation=.98,
        coord_frame=coord_frame)
    return raw


def _parse_bad_channels(sss_log):
    with open(sss_log) as fid:
        bad_lines = {l for l in fid.readlines() if 'Static bad' in l}
    bad_channels = list()
    for line in bad_lines:
        chans = line.split(':')[1].strip(' \n').split(' ')
        for cc in chans:
            ch_name = 'MEG%01d' % int(cc)
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
    return bad_channels


def _process_subject(subject, kind):
    meg_inp_dir = op.join(data_path, kind)
    raw = mne.io.read_raw_fif(
        op.join(meg_inp_dir, '{}'.format(subject),
                '{}_raw.fif'.format(kind)))

    if kind == 'emptyroom':
        kind_ = 'rest'
        coord_frame = 'meg'
        meg_inp_dir = op.join(data_path, 'rest')

    else:
        kind_ = kind
        coord_frame = 'head'

    sss_log = op.join(meg_inp_dir, '{}'.format(subject),
                      'mf2pt2_{}_raw.log'.format(kind_))
    raw.info['bads'] = _parse_bad_channels(sss_log)

    raw.load_data()
    raw_sss = _run_maxfilter(raw, coord_frame)
    return raw, raw_sss, subject, kind


out = (_process_subject(
    subject=subject, kind=kind) for subject in subjects for kind in kinds)

for (raw, raw_sss, subject, kind) in out:

    raw_sss.save(
        op.join(data_path, 'meg_dir', subject, '{}-sss-raw.fif'.format(kind)),
        overwrite=True)

    fig = raw.plot_psd(xscale='log')
    fig.savefig('./figures_qc/{}-{}-raw.png'.format(subject, kind), dpi=150)

    fig = raw_sss.plot_psd(xscale='log')
    fig.savefig('./figures_qc/{}-{}-raw_sss.png'.format(subject, kind),
                dpi=150)
    plt.close('all')
