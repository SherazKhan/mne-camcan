import mne
import os.path as op
import matplotlib.pyplot as plt
from camcan.library.config import ctc, cal
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import glob
import numpy as np
from scipy.signal import hilbert
from scipy.stats import ranksums, ttest_ind
from camcan.utils import get_stc, make_surrogates_empty_room
import bct
from joblib import Parallel, delayed
from camcan.utils import distcorr
mne.set_log_level('WARNING')
import joblib
import os
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import time
import datetime
from mne.filter import resample, filter_data
from mne.label import read_labels_from_annot
plt.ion()
subjects = ['CC110033', 'CC110037', 'CC110045']
subject = subjects[0]

# For Desktop
data_path = '/cluster/transcend/sheraz/Dropbox/mne-camcan-data/'

subjects_dir = op.join(data_path,'recons')
subject_dir = op.join(subjects_dir,subject)
bem_dir = op.join(subject_dir,'bem')
trans_file = op.join(data_path, 'trans',subject + '-trans.fif')
labels_fname  = glob.glob(op.join(data_path, 'labels', '*.label'))
labels = [mne.read_label(label, subject='fsaverageSK', color='r')
          for label in labels_fname]
# labels =read_labels_from_annot('fsaverage', 'aparc', 'both', subjects_dir=subjects_dir)
# labels.pop(-1)
for index, label in enumerate(labels):
    label.values.fill(1.0)
    labels[index] = label
mne.set_config('SUBJECTS_DIR',subjects_dir)
labels = [label.morph('fsaverageSK', subject, subjects_dir=subjects_dir) for label in labels]

event_id = 1
event_overlap = 8
event_length = 30
spacing='ico5'
lf, hf = 14., 30.

#def process_maxfilter(subject):
raw_fname = op.join(
    data_path, 'rest', 'sub-' + subject,'meg', 'rest_raw.fif')

erm_fname = op.join(
    data_path, 'emptyroom', '090430_raw.fif')


raw = mne.io.read_raw_fif(raw_fname, preload=True)
erm_raw = mne.io.read_raw_fif(erm_fname, preload=True)

raw_length = (raw.last_samp-raw.first_samp)/raw.info['sfreq']
raw.info['bads'] +=  [u'MEG2113', u'MEG1941', u'MEG1412', u'MEG2331']

erm_raw_length = (erm_raw.last_samp-erm_raw.first_samp)/erm_raw.info['sfreq']
erm_raw.info['bads'] +=  [u'MEG2113', u'MEG2112', u'MEG0512', u'MEG0141', u'MEG2533']
raw.fix_mag_coil_types()
erm_raw.fix_mag_coil_types()

raw = mne.preprocessing.maxwell_filter(raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       origin=(0., 0., 0.04))

erm_raw = mne.preprocessing.maxwell_filter(erm_raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       coord_frame='meg',
                                       origin=(0., 0.013, -0.006))

projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, eog_events = compute_proj_eog(raw, n_grad=1, n_mag=2)

raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog

erm_raw.info['projs'] += projs_ecg
erm_raw.info['projs'] += projs_eog

raw.apply_proj()
erm_raw.apply_proj()


raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

erm_raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

erm_cov = mne.compute_raw_covariance(erm_raw, tmin=0, tmax=None)

raw.filter(lf, hf, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')
reject = dict(grad=1000e-13, mag=1.2e-12)
events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap, start=0, stop=raw_length-event_length)
epochs = mne.Epochs(raw, events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
epochs.resample(hf*3)



bem_fname = op.join(bem_dir, '%s-src.fif' % subject)
src_fname = op.join(bem_dir, '%s-src.fif' % spacing)

bem = mne.read_bem_solution(bem_fname)
src = mne.read_source_spaces(src_fname)

fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, n_jobs=2)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, erm_cov,loose=0.2, depth=0.8)

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
projected_erm_raw = make_surrogates_empty_room(erm_raw, fwd_fixed, inv, step=10000)


projected_erm_raw.filter(lf, hf, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')
projected_erm_events = mne.make_fixed_length_events(projected_erm_raw, event_id,
                                                    duration=event_overlap, start=0,
                                                    stop=erm_raw_length-event_length)
projected_erm_epochs = mne.Epochs(projected_erm_raw, projected_erm_events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
projected_erm_epochs.resample(hf*3)

counter = []
for index1 in range(len(labels)-1):
    for index2 in range(index1+1,len(labels)):
        counter.append((index1, index2))




def distcorr_n(X, Y):

    n = X.shape[0]

    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor

# def compute_zdistcor(projected_erm_epochs, epochs, labels, index1, index2, method='MNE', snr=1):
#
#     lambda2 = 1.0 / snr ** 2
#     stcs = mne.minimum_norm.apply_inverse_epochs(projected_erm_epochs, inv, lambda2, method, labels[index1], pick_ori="normal")
#     data_label1 = np.abs(hilbert(np.transpose(np.array([stc.data for stc in stcs]), (0, 2, 1)), axis=1))
#
#     stcs = mne.minimum_norm.apply_inverse_epochs(projected_erm_epochs, inv, lambda2, method, labels[index2], pick_ori="normal")
#     data_label2 = np.abs(hilbert(np.transpose(np.array([stc.data for stc in stcs]), (0, 2, 1)), axis=1))
#
#     corr_erm = np.abs(np.array([distcorr(l1, l2, 0) for l1, l2 in zip(data_label1, data_label2)]))
#
#     stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, labels[index1], pick_ori="normal")
#     data_label1= np.abs(hilbert(np.transpose(np.array([stc.data for stc in stcs]), (0, 2, 1)), axis=1))
#
#     stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, labels[index2], pick_ori="normal")
#     data_label2 = np.abs(hilbert(np.transpose(np.array([stc.data for stc in stcs]), (0, 2, 1)), axis=1))
#
#     corr_rest = np.abs(np.array([distcorr(l1, l2, 0) for l1, l2 in zip(data_label1, data_label2)]))
#     print(ttest_ind(corr_rest, corr_erm)[0])
#     return ttest_ind(corr_rest, corr_erm)[0]


def compute_zdistcor_resample(projected_erm_epochs, epochs, labels, index1, index2, method='MNE', snr=1):

    lambda2 = 1.0 / snr ** 2
    stcs = mne.minimum_norm.apply_inverse_epochs(projected_erm_epochs, inv, lambda2, method, labels[index1], pick_ori="normal")
    data_label1 = np.abs(hilbert(np.array([stc.data for stc in stcs]), axis=2))
    data_label1 = filter_data(data_label1, projected_erm_epochs.info['sfreq'], None, 2, fir_design="firwin2")
    data_label1 = np.transpose(resample(data_label1, down=0.25*projected_erm_epochs.info['sfreq'], axis=2, npad='auto'), (0,2,1))[:,5:-5,:]

    stcs = mne.minimum_norm.apply_inverse_epochs(projected_erm_epochs, inv, lambda2, method, labels[index2], pick_ori="normal")
    data_label2 = np.abs(hilbert(np.array([stc.data for stc in stcs]), axis=2))
    data_label2 = filter_data(data_label2, projected_erm_epochs.info['sfreq'], None, 2, fir_design="firwin2")
    data_label2 = np.transpose(resample(data_label2, down=0.25*projected_erm_epochs.info['sfreq'], axis=2, npad='auto'), (0,2,1))[:,5:-5,:]

    corr_erm =np.abs(np.array([distcorr_n(l1, l2) for l1, l2 in zip(data_label1, data_label2)]))

    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, labels[index1], pick_ori="normal")
    data_label1 = np.abs(hilbert(np.array([stc.data for stc in stcs]), axis=2))
    data_label1 = filter_data(data_label1, epochs.info['sfreq'], None, 2, fir_design="firwin2")
    data_label1 = np.transpose(resample(data_label1, down=0.25*epochs.info['sfreq'], axis=2, npad='auto'), (0,2,1))[:,5:-5,:]

    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, labels[index2], pick_ori="normal")
    data_label2 = np.abs(hilbert(np.array([stc.data for stc in stcs]), axis=2))
    data_label2 = filter_data(data_label2, epochs.info['sfreq'], None, 2, fir_design="firwin2")
    data_label2 = np.transpose(resample(data_label2, down=0.25*epochs.info['sfreq'], axis=2, npad='auto'), (0,2,1))[:,5:-5,:]

    corr_rest = np.abs(np.array([distcorr_n(l1, l2) for l1, l2 in zip(data_label1, data_label2)]))

    return ranksums(corr_rest, corr_erm)[0]



#from pyimpress.utils import ParallelExecutor

def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time()-tick
        avg_speed = time_diff/step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)

all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}

def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun


n_jobs = 235
aprun = ParallelExecutor(n_jobs=n_jobs)
corr_z = aprun(total=len(counter))(delayed(compute_zdistcor_resample)(projected_erm_epochs, epochs, labels, index[0], index[1]) for index in counter)






# corr_z = Parallel(n_jobs=8, verbose=5)(delayed(compute_zdistcor)(projected_erm_epochs, epochs, labels, index[0], index[1]) for index in tqdm(counter))
#
# corr_z = np.zeros((len(counter),))
# for ind, index in enumerate(counter):
#     #print('%.4f Percent Done'% (float(ind+1)/float(len(counter))*100))
#     corr_z[ind] = compute_zdistcor(projected_erm_epochs, epochs, labels, index[0], index[1])

#
data = {'corr_z': corr_z, 'labels':labels, 'counter':counter}

pkl_fname = os.path.join(data_path,subject + '_lf_' + str(int(lf)) + '_hf_' + str(int(hf))+
                         '_labels_' + str(len(labels)) + '_timestamp_'+ '_'.join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(' '))
 +'_corr_z.pkl')
joblib.dump(data, pkl_fname)

#
#
#
corr_zz =  np.zeros((len(labels), len(labels)))
for index in range(len(counter)):
        corr_zz[counter[index]] = corr_z[index]


corr_zz = corr_zz + corr_zz.T

corr = np.int32(bct.utils.threshold_proportional(corr_zz,.15) > 0)
deg = np.array(bct.degrees_und(corr))

stc = get_stc(labels_fname, deg)
brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                  surface='inflated10', subjects_dir=subjects_dir)
#
# brain.save_image('beta_orthogonal_corr.png')



# def exact_mc_perm_test(xs, ys, nmc=10000):
#     n, k = len(xs), 0
#     diff = np.abs(ttest_ind(xs,ys)[0])
#     zs = np.concatenate([xs, ys])
#     for j in range(nmc):
#         np.random.shuffle(zs)
#         k += diff < np.abs(ttest_ind(zs[:n], zs[n:])[0])
#     return k / float(nmc)


#
# pkl_fname = os.path.join(data_path, 'CC110033_lf_8_hf_11_labels_154_corr_z.pkl')
# x = joblib.load(pkl_fname)
# counter = x['counter']
# labels = x['labels']
# corr_z = x['corr_z']
#
# corr_zz =  np.zeros((len(labels), len(labels)))
# for index in range(len(counter)):
#     corr_zz[counter[index]] = corr_z[index]
# corr_zz = corr_zz + corr_zz.T
# corr = np.int32(bct.utils.threshold_proportional(corr_zz,.15) > 0)
# deg = np.array(bct.degrees_und(corr))
# stc = get_stc(labels_fname, deg)
# # #
# brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
#                   views=['lateral','medial'],
#                  surface='inflated10', subjects_dir=subjects_dir, clim={'kind':'value', 'lims':(5, 10, 25)})

























































