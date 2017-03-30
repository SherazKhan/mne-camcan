import mne
import os
from scipy.io import loadmat, savemat
import pyimpress as pyi
import numpy as np
import joblib
from mne.parallel import parallel_func
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import joblib

plt.ion()

dir_path = os.path.join('/'.join(os.path.dirname(pyi.__file__).split('/')[:-1]),'data')
labels_path = '/cluster/transcend/sheraz/NC_rev/labels/'
hkl_fname = os.path.join(dir_path,'labels.hkl')
labels = joblib.load(hkl_fname)['labels']

temp =loadmat(os.path.join(dir_path,'sub_fix.mat'))['subj']
subjs = np.array([(str(t[0])[3:-2],int(str(t[1])[2:-2]),int(str(t[2])[2:-2])) for t in temp])


def compute_crosstalk_maps(subj,labels):

    if subj[0][:3] not in 'nmr':
        fwd_fname ='/autofs/cluster/transcend/MEG/fix/' + subj[0] + '/X/' +subj[0] + '_fix_1-fwd.fif'
        inv_fname = '/autofs/cluster/transcend/MEG/fix/' + subj[0] + '/X/' + subj[0] + '_fix_144fil_proj-inv.fif'
        if not os.path.isfile(inv_fname):
            inv_fname = '/autofs/cluster/transcend/MEG/fix/' + subj[0] + '/X/' + subj[
                0] + '_fix_1_144_fil_fixed_new_erm_megreg_0_new_MNE_proj-inv.fif'
        labels_s = [mne.read_label(labels_path + subj[0] + '/' + subj[0] + '-' + label) for label in labels]
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        stc_ctf = mne.minimum_norm.cross_talk_function(inv, fwd, labels_s, method='MNE')



    else:
        fwd_fname = '/autofs/cluster/fusion/Sheraz/rs/Linda/'+ subj[0] + '/rest01_bad-ico-5-meg-fwd.fif'
        inv_fname = '/autofs/cluster/fusion/Sheraz/rs/Linda/' + subj[0] + '/rest01_bad-ico-5-B-meg-inv.fif'
        labels_s = [mne.read_label(labels_path + subj[0] + '_FS/' + subj[0] + '_FS-' + label) for label in labels]
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        stc_ctf = mne.minimum_norm.cross_talk_function(inv, fwd, labels_s, method='MNE')

    return stc_ctf



parallel, run_func, _ = parallel_func(compute_crosstalk_maps, n_jobs=24)
crosstalk_maps = parallel(run_func(subj,labels) for index, subj in enumerate(subjs))

ages = subjs[:,1].astype(np.int)
data = {'subjs': subjs, 'crosstalk_maps':crosstalk_maps, 'ages':ages}

hkl_fname = os.path.join(dir_path,'sub_fix_crosstalk.hkl')
hkl.dump(data, hkl_fname)

crc = np.array([np.median((stc.data[:,-1]/max(stc.data[:,-1]))) for stc in crosstalk_maps])

ind_redo = np.where(crc > 0.105)[0]

subjs_r = subjs[ind_redo]

parallel, run_func, _ = parallel_func(compute_crosstalk_maps, n_jobs=16)
crosstalk_maps_r = parallel(run_func(subj,labels) for index, subj in enumerate(subjs_r))
crosstalk_maps = np.array(crosstalk_maps)
crosstalk_maps_r = np.array(crosstalk_maps_r)
crosstalk_maps[ind_redo] = crosstalk_maps_r

crc = np.array([np.mean((stc.data[:,-1]/max(stc.data[:,-1]))) for stc in crosstalk_maps])


def morph_stc(stc):
    try:
        stc = stc.morph('fsaverage', grade=5, smooth=5)
    except:
        pass
    return stc

parallel, run_func, _ = parallel_func(morph_stc, n_jobs=28)
crosstalk_morph_maps = parallel(run_func(stco) for stco in crosstalk_maps)

ind = np.arange(len(crosstalk_morph_maps))

crc = np.array([np.mean((stc.data[:,-1]/max(stc.data[:,-1]))) for stc in crosstalk_morph_maps])

ids = subjs[:,0].astype(np.str)

ind_nmr = np.hstack((np.array(mne.pick_channels_regexp(ids, 'nmr*')),130))

ind_all= np.arange(len(crosstalk_morph_maps))
ind_good = np.delete(ind_all,ind_nmr)


9|import| |glob
15|#labels = [os.path.basename(label) for label in |glob|.glob('/cluster/transcend/sheraz/NC_rev/labels/*.label')]
15|#labels = [os.path.basename(label) for label in glob.|glob|('/cluster/transcend/sheraz/NC_rev/labels/*.label')]

% %
function
corr = hippcorr_f(x, cfg, band)

nTime = size(x, 2);
nEpoch = size(x, 3);
nROI = size(x, 1);
corr = cell(1);

for i=1:length(band)
x1 = eegfilt(double(x), cfg.fs, band
{i}.hp, band
{i}.lp, nTime);


x1 = reshape(x1, nROI, nTime, nEpoch);

y1 = zeros(nROI, nTime, nEpoch);

for j=1:nEpoch
y1(:,:, j)=hilbert(x1(:,:, j)')';
end

co = zeros(nROI, nROI, nEpoch);
fs = cfg.fs;

load('/autofs/cluster/transcend/sheraz/scripts/filtwts.mat');

for ii = 1:nROI
y_orth = (imag((permute(repmat(squeeze(y1(ii,:,:)), [1 1 nROI]), [3 1 2])).*(conj(y1). / abs(y1))));
y_orig = abs(y1(ii,:,:));
y = cat(1, y_orig, y_orth);

times = 0:1 / 256:30;
[y, times] = compute_resample_empty(y, times);
nTime1 = size(y, 2);

y = permute(reshape(filtfilt(filtwts, 1, reshape(permute(y, [2 1 3]), nTime1, (nROI + 1) * nEpoch)), nTime1, (nROI + 1),
                    nEpoch), [2 1 3]);

co_temp = zeros(nROI + 1, nROI + 1, nEpoch);
for jj = 1:nEpoch

co_temp(:,:, jj)=corrcoef((y(:, 7:end - 7, jj))');
end

co(ii,:,:)=squeeze(co_temp(1, 2:end,:));

end

corr
{i}.co = co;

corr
{i}.hp = band
{i}.hp;
corr
{i}.lp = band
{i}.lp;

end