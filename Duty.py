# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne import read_evokeds
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
from autoreject import get_rejection_threshold
from autoreject import AutoReject
from mne.preprocessing import find_bad_channels_maxwell
from IPython import get_ipython

# %%
import mne
mne.sys_info()


# %% [markdown]
# https://mne.tools/stable/auto_tutorials/intro/plot_10_overview.html#sphx-glr-auto-tutorials-intro-plot-10-overview-py
# %% [markdown]
# ## Data importation

# %%
subject = 'CC110033'
sample_data_raw_file = os.path.join(subject,
                                    'task',
                                    'task_raw.fif')

sample_data_raw_file_er = os.path.join(
    "emptyroom",
    subject,
    'emptyroom_' + subject + '.fif')


# %%
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw_er = mne.io.read_raw_fif(sample_data_raw_file_er)

# %% [markdown]
# here are 8 “projection items” in the file along with the recorded data;
#  those are SSP projectors calculated to remove environmental noise from
#  the MEG signals, plus a projector to mean-reference the EEG channels;
# these are discussed in the tutorial Background on projectors and projections.

# %%
print(raw)
raw.info

# %% [markdown]
# interpretations :
# 339 channels, 147000 points per record, 1000 Hz.
# 204 GRAD, 102 MAG, 17 STIM, 2 EOG, 1 ECG, 13 MIS > what is this.

# %%
raw_er.info

# %% [markdown]
# ### PSD

# %%
_ = raw.plot_psd(fmax=50)
_ = raw.plot(duration=5, n_channels=30)


# %%
_ = raw_er.plot_psd(fmax=50)
_ = raw_er.plot(duration=5, n_channels=30)

# %% [markdown]
# ## Preprocessing

# %%
# fine_cal_file = os.path.join("/content/drive/MyDrive/PFE/neurospin/welcome_duty",
#                              "welcome_duties_sss_cal.dat")
fine_cal_file = os.path.join(
    "welcome_duties_sss_cal.dat")

# Is there a crosstalk file anywhere ?
# crosstalk_file = os.path.join(sample_data_folder, 'SSS', 'ct_sparse_mgh.fif')

# %% [markdown]
# ### find_bad_channels_maxwell

# %%


def find_bad_channels_maxwell_util(raw):
    raw.info['bads'] = []
    raw_check = raw.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check,
        # cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        return_scores=True, verbose=True, coord_frame="meg")

    print("auto_noisy_chs", auto_noisy_chs)
    print("auto_flat_chs", auto_flat_chs)
    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads
    raw.auto_scores = auto_scores


# %%
find_bad_channels_maxwell_util(raw)

# %% [markdown]
# Copy paste is evil, isn't there a way to create a pipeline like in Sklearn ?

# %%
find_bad_channels_maxwell_util(raw_er)

# %% [markdown]
# Union of bads for empty room and not empty room ?

# %%


def union_bads(raw, raw_er):
    """ Update bads of raw and raw_er
    Make the union of bads.
    """
    print(set(raw.info['bads']), set(raw_er.info['bads']))
    all_bads = set(raw.info['bads']) | set(raw_er.info['bads'])
    all_bads = list(all_bads)
    raw.info['bads'] = all_bads
    raw_er.info['bads'] = all_bads


union_bads(raw, raw_er)

# %% [markdown]
# ### manual inspection : diagnostic figures

# %%


def manual_inspection(raw):
    auto_scores = raw.auto_scores
    # Only select the data for gradiometer channels.
    ch_type = 'grad'
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                  for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# %% [markdown]
# #### raw


# %%
manual_inspection(raw)

# %% [markdown]
# The channel 2043 seems very noisy.
# And the channels 0933, 1212, 2023 are suspicous. Better not let pass anything.

# %%
# Check of channel 2043 and 2023
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG20.')
raw.plot(order=picks, n_channels=len(picks))


# %%
# Check of channels 1212 and 0933
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG121.|MEG093.')
raw.plot(order=picks, n_channels=len(picks))


# %%
# from manual inspection
raw.info['bads'] += ['MEG2043', 'MEG0933', 'MEG1212', 'MEG2023']

# %% [markdown]
# #### raw_empty_room

# %%
manual_inspection(raw_er)

# %% [markdown]
# The channel 2043 seems very noisy.
# And the channels 0933, 1212, 2023 are suspicous. Better not let pass anything.

# %%
# Check of channel 0913
picks = mne.pick_channels_regexp(raw_er.ch_names, regexp='MEG09.')
raw_er.plot(order=picks, n_channels=len(picks))


# %%
# from manual inspection
raw_er.info['bads'] += ['MEG0913']


# %%
union_bads(raw, raw_er)

# %% [markdown]
# ## maxwell_filter

# %%


def maxwell_filter_utils(raw):
    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        # cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        verbose=True,
        coord_frame="meg"  # Not sure about this parameter
    )
    raw.copy().pick(['meg']).plot(duration=2, butterfly=True)
    raw_sss.copy().pick(['meg']).plot(duration=2, butterfly=True)
    return raw_sss

# %% [markdown]
# #### raw


# %%
raw_sss = maxwell_filter_utils(raw)

# %% [markdown]
# #### raw_er

# %%
raw_sss_er = maxwell_filter_utils(raw_er)

# %% [markdown]
# You can plot the PSD to have a first look at the signals and see the effects
# of the filtering.
# This tutorial might be useful for understanding this task. :
# https://mne.tools/0.11/auto_examples/time_frequency/plot_compute_raw_data_spectrum.html ?
# %% [markdown]
# ## Bandpass filter
# %% [markdown]
# #### raw

# %%
raw.plot_psd(area_mode='range', tmax=10.0, picks=None, average=False)

# %% [markdown]
# Why frequencies = 50, 340 are dashed ?
# Answer : The line noise frequency is
# also indicated with a dashed line (⋮) so probably frequendcies after 340 are
#  spotted as noise by mne and 50 was also spotted by mne.
#
#  Why a pair of identical plot ?
# %% [markdown]
# RuntimeError: By default, MNE does not load data into main memory to conserve
# resources. inst.filter requires raw data to be loaded. Use preload=True (or string)
# the constructor or raw.load_data().

# %%
raw.load_data()
raw.filter(l_freq=None, h_freq=40)


# %%
raw.plot_psd(area_mode='range', tmax=10.0, picks=None, average=False)

# %% [markdown]
# #### raw_er

# %%
raw_er.plot_psd(area_mode='range', tmax=10.0, picks=None, average=False)


# %%
raw_er.load_data()
raw_er.filter(l_freq=None, h_freq=40)


# %%
raw_er.plot_psd(area_mode='range', tmax=10.0, picks=None, average=False)

# %% [markdown]
# ## Construct epochs from MEG data

# %%
event_name_to_id_mapping = {
    'audiovis/300Hz': 1,
    'audiovis/600Hz': 2,
    'audiovis/1200Hz': 3,
    'catch/0': 4,
    'catch/1': 5,
    'audio/300Hz': 6,
    'audio/600Hz': 7,
    'audio/1200Hz': 8,
    'vis/checker': 9,
    'button': 99}


# %%
def create_events_epochs(raw):
    events = mne.find_events(raw, min_duration=0.001, shortest_event=2)
    epochs = mne.Epochs(raw, events, event_id=event_name_to_id_mapping,
                        on_missing='ignore')

    for (e, i) in event_name_to_id_mapping.items():
        a = (events[:, -1] == i).sum()
        print(f"event {e} is present {a} times")

    raw.events = events
    raw.epochs = epochs


# %%
create_events_epochs(raw)


# %%
# I think there is no notion of event for an empty room


# %%
events_id = {
    'audio/1200Hz': 0,
    'audio/300Hz': 3,
    'audio/600Hz': 0,
    'audiovis/1200Hz': 40,
    'audiovis/300Hz': 40,
    'audiovis/600Hz': 37,
    'button': 0,
    'catch/0': 4,
    'catch/1': 4}

# %% [markdown]
# ##  use autoreject local to clean the data from remaining artifacts.
#
#
#

# %%
ar = AutoReject()

epochs = raw.epochs
epochs.load_data()
# epochs_clean = ar.fit_transform(epochs,)


# %%
reject = get_rejection_threshold(epochs)


# %%
reject

# %% [markdown]
# Is this related with the bonferroni correction ?
# %% [markdown]
# ## Compute evoked responses for the visual and auditory events.
# When does the signal peaks? Does the topographies look dipolar at the peak latencies?
# %% [markdown]
# ### evoked responses for the visual and auditory events

# %%
events_id


# %%
evoked = epochs_clean['audiovis/1200Hz'].average()
evoked.plot()

# %% [markdown]
# ### When does the signal peaks? Does the topographies look dipolar at the peak
# latencies?
#
# - peak at 0.13s, which seems appropriate.

# %%
times = np.arange(0.05, 0.251, 0.04)
evoked.plot_topomap(times, ch_type='mag', average=0.05, time_unit='s')


# %%
fig = evoked.plot_topomap(times, ch_type='grad', average=0.05, time_unit='s')

# %% [markdown]
# Does the topographies look dipolar at the peak latencies?
# > It is clearly dipolar, the response is way more strong on the left.
# %% [markdown]
# ## Compute the noise covariance from the baseline segments with optimal shrinkage
# and document the quality of the resulting spatial - whitening

# %%
raw_er.info['bads'] = [
    bb for bb in raw.info['bads'] if 'EEG' not in bb]
raw_er.add_proj(
    [pp.copy() for pp in raw.info['projs'] if 'EEG' not in pp['desc']])

noise_cov = mne.compute_raw_covariance(
    raw_er, tmin=0, tmax=None)

# %%
# We also use the pre-stimulus baseline to estimate the noise covariance
# epochs = mne.Epochs(raw, events, event_id=event_name_to_id_mapping,
#                         on_missing='ignore')

ante_epochs = mne.Epochs(raw, raw.events, event_id=1, tmin=-0.2, tmax=0.5,
                         # we'll decimate for speed
                         baseline=(-0.2, 0.0), decim=3,
                         verbose='error')  # and ignore the warning about aliasing

noise_cov_baseline = mne.compute_covariance(ante_epochs, tmax=0)

# %%
noise_cov.plot(raw_er.info, proj=True)
noise_cov_baseline.plot(ante_epochs.info, proj=True)

# %% ?
subjects_dir = os.path.join("freesurfer")


mne.gui.coregistration(tabbed=None, split=True, width=None,
                       inst=sample_data_raw_file,
                       subject=subject, subjects_dir=subjects_dir,
                       guess_mri_subject=None, height=None,
                       head_opacity=None, head_high_res=None, trans=None,
                       scrollable=True,
                       project_eeg=None, orient_to_surface=None,
                       scale_by_distance=None,
                       mark_inside=None, interaction=None, scale=None,
                       advanced_rendering=None, verbose=None)

# %%
trans_dir = os.path.join("trans", "sub-" + subject + "-trans.fif")
# mne.viz.plot_alignment(info=None, trans=trans_dir, subject=subject,
#                        subjects_dir=subjects_dir, surfaces='auto',
#                        coord_frame='head', meg=None,
#                        eeg='original', fwd=None, dig=False,
#                        ecog=True, src=None,
#                        mri_fiducials=False, bem=None, seeg=True,
#                        fnirs=True, show_axes=False, fig=None,
#                        interaction='trackball', verbose=None)


# %% [Markdown] Forward problem
# %%
#
bem_dir = os.path.join("freesurfer",
                       subject,
                       "bem",
                       subject+"-meg-bem.fif")  # or maybe "-head.fif"


subjects_dir = os.path.join("freesurfer")

# surfs = mne.make_bem_model(subject, ico=4,
#                          conductivity=(0.3, 0.006, 0.3),
#                          subjects_dir=subjects_dir, verbose=None)
#
# bem = mne.make_bem_solution(surfs, verbose=None)

src = mne.setup_source_space(subject, spacing='oct5',  # oct6
                             add_dist=False, subjects_dir=subjects_dir)  # subjects_dir ?

fwd = mne.make_forward_solution(info=raw.info, trans=trans_dir,
                                src=src, bem=bem_dir,  # or bem ?
                                meg=True, eeg=True, mindist=0.0,
                                ignore_ref=False, n_jobs=1,
                                verbose=None)

# %%

inv = mne.minimum_norm.make_inverse_operator(info=raw.info,
                                             forward=fwd,
                                             noise_cov=noise_cov,
                                             loose='auto', depth=0.8,
                                             fixed='auto', rank=None,
                                             use_cps=True, verbose=None)
# src = inverse_operator['src']

# Compute inverse solution
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

stc = apply_inverse(evoked, inv, lambda2, method)
stc.crop(0.0, 0.2)

# Export result as a 4D nifti object
img = stc.as_volume(src,
                    mri_resolution=False)  # set True for full MRI resolution

# Plotting with nilearn ######################################################
plot_stat_map(index_img(img, 61), t1_fname, threshold=8.,
              title='%s (t=%.1f s.)' % (method, stc.times[61]))


# %%
# ICA to remove eye-blinks and cardiac artifacts

filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=None)
ica.fit(filt_raw)
