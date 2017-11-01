import mne
import glob
import timeit
import numpy as np
from dask import delayed
from dask.distributed import LocalCluster, Client
from mne.time_frequency import psd_multitaper
import datetime
import os
import matplotlib.pyplot as plt

# Reduce Verbosity
mne.set_log_level('WARNING')
# Numbers of subjects to be processed in parallel
total_workers = range(5, 36, 5)
# For sanity only first 200 subject out of 650



fif_files_path = '/autofs/cluster/fusion/Sheraz/data/camcan/camcan47' '/cc700/meg/pipeline/release004/data_nomovecomp' '/aamod_meg_maxfilt_00001/*/rest/transdef_mf2pt2_rest_raw.fif'

files = glob.glob(fif_files_path)

print(len(files))


def compute_psd(fif_file):
    raw = mne.io.read_raw_fif(fif_file, preload=False)
    raw.crop(50, 54)
    picks = mne.pick_types(raw.info, meg='mag', eeg=False,
                           eog=False, stim=False)
    psd, _ = psd_multitaper(raw, fmin=2, fmax=55, picks=picks, normalization="full")
    return np.log10(psd)


time_elpased_lparallel = []

for workers in total_workers:

    cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
    client = Client(cluster)
    psds = []
    print(workers)
    for file in files[0:35]:
        psd = delayed(compute_psd)(file)
        psds.append(psd)

    mean_psd_delayed = delayed(np.mean)(psds, axis=1)

    start_time = timeit.default_timer()
    mean_psd = client.compute(mean_psd_delayed)
    mean_psd = mean_psd.result()
    time_elpased_lparallel.append(timeit.default_timer() - start_time)
    print(time_elpased_lparallel)
    client.close()
    cluster.close()

plt.plot(total_workers, time_elpased_lparallel)

host_name = os.uname()[1].split('.')[0] +"_"
time_tag = "_".join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(' '))

np.savez(host_name + time_tag + '.npz', time_elpased_lparallel=time_elpased_lparallel, total_workers=total_workers)