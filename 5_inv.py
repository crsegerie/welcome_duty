""" This file compute the inverse :
    - compute the Sources src
    - Compute the forward problem
    - Compute the inverse

"""
# %%
import mne
from mne.minimum_norm import apply_inverse

from params import subjects_dir, subject, trans_dir
from params import raw_maxfiltered_file, evoked_file, noise_cov_baseline_file

raw = mne.io.read_raw_fif(raw_maxfiltered_file)
noise_cov = mne.read_cov(noise_cov_baseline_file)
evoked = mne.read_evokeds(evoked_file)[0]

# %% Forward problem

src = mne.setup_source_space(subject, spacing='oct5',  # oct6
                             add_dist=False, subjects_dir=subjects_dir)  # subjects_dir ?

bem = mne.make_bem_model(subject, ico=4,
                         subjects_dir=subjects_dir, verbose=None)

bem_sol = mne.make_bem_solution(surfs=bem, verbose=None,)

fwd = mne.make_forward_solution(info=raw.info, trans=trans_dir,
                                src=src, bem=bem_sol,
                                meg=True, eeg=True, mindist=0.0,
                                ignore_ref=False, n_jobs=1,
                                verbose=None)

inv = mne.minimum_norm.make_inverse_operator(info=raw.info,
                                             forward=fwd,
                                             noise_cov=noise_cov,
                                             loose='auto', depth=0.8,
                                             fixed='auto', rank=None,
                                             use_cps=True, verbose=None)

# Compute inverse solution
snr = 3.0
lambda2 = 1.0 / snr ** 2  # TODO: understand this mystery
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

stc = apply_inverse(evoked, inv, lambda2, method)
stc.crop(0.0, 0.2)
