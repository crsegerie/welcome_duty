"""
- Import and create raw files.
- Visualisation of raw files.
- Maxfilter :
    - bad auto check
    - bad manual check
    - Maxfilter
- Plot PSD
- Band Pass data
"""

# TODO : this file is ugly
# %%
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import find_bad_channels_maxwell

from params import (sample_data_raw_file, sample_data_raw_file_er,
                    fine_cal_file, h_freq,
                    raw_maxfiltered_file, raw_er_maxfiltered_file)


# %%
# Import and create raw files + Visualise ########################
##################################################################

def import_raws(verbose=False, plot=False):
    ''' Import raw and raw of empty room.
    returns raw, raw_er
    '''
    raw_ = mne.io.read_raw_fif(sample_data_raw_file)
    raw_er_ = mne.io.read_raw_fif(sample_data_raw_file_er)
    if verbose:
        print(raw_.info)
        print(raw_er_.info)
    if plot:
        _ = raw_.plot_psd(fmax=50)
        _ = raw_.plot(duration=5, n_channels=30)
        _ = raw_er_.plot_psd(fmax=50)
        _ = raw_er_.plot(duration=5, n_channels=30)
    return raw_, raw_er_


# %%
# # BUG filter from 22.5s
raw, raw_er = import_raws(verbose=True, plot=True)
# interpretations :
# 339 channels, 147000 points per record, 1000 Hz.
# 204 GRAD, 102 MAG, 17 STIM, 2 EOG, 1 ECG, 13 MIS > what is this.
# Answer : Gradi = gradiometers, MAG = MEG , STIM = event, EOG = ocular,
# ECG = cardio, mis ?


# %%
# Find bad channels Automatically  ###############################
##################################################################

def find_bad_channels_maxwell_util(raw_):
    """ Inplace update of bad channels.
    Find noisy and flat channels according to the Maxwell filter.
    """
    raw_.info['bads'] = []
    raw_check = raw_.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check,
        # cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        return_scores=True, verbose=True, coord_frame="meg")

    print("auto_noisy_chs", auto_noisy_chs)
    print("auto_flat_chs", auto_flat_chs)
    bads = raw_.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw_.info['bads'] = bads
    raw_.auto_scores = auto_scores


def union_bads():
    """ inplace update  bads of raw and raw_er
    Make the union of bads.
    """
    print(set(raw.info['bads']), set(raw_er.info['bads']))
    all_bads = set(raw.info['bads']) | set(raw_er.info['bads'])
    all_bads = list(all_bads)
    raw.info['bads'] = all_bads
    raw_er.info['bads'] = all_bads


find_bad_channels_maxwell_util(raw)
find_bad_channels_maxwell_util(raw_er)
union_bads()

# %%
# Manual Inspection of bad Channels ##############################
##################################################################


def manual_inspection(raw_):
    ''' Return a heatmap allowing to inspect noisy channels.
    '''
    auto_scores = raw_.auto_scores
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

    # First, plot the "raw_" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


manual_inspection(raw)
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


# %%
# Manual Inspection of bad Channels in the Empty room ############
##################################################################

manual_inspection(raw_er)

# %% [markdown]
# The channel 2043 seems very noisy.
# And the channels 0933, 1212, 2023 are suspicous.
# Better not let pass anything.

# %%
# Check of channel 0913
picks = mne.pick_channels_regexp(raw_er.ch_names, regexp='MEG09.')
raw_er.plot(order=picks, n_channels=len(picks))
raw_er.info['bads'] += ['MEG0913']
union_bads()


# %%
# Maxwell Filter after having marked the bad channels ############
##################################################################

def maxwell_filter_utils(raw_):
    '''From the raw_ file, compute and returns maxfiltered raw_sss_ file.'''
    raw_sss_ = mne.preprocessing.maxwell_filter(
        raw_,
        # cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        verbose=True,
        coord_frame="meg"  # Not sure about this parameter
    )
    raw_.copy().pick(['meg']).plot(duration=2, butterfly=True)
    raw_sss_.copy().pick(['meg']).plot(duration=2, butterfly=True)
    return raw_sss_


raw_sss = maxwell_filter_utils(raw)
raw_sss_er = maxwell_filter_utils(raw_er)

# %%
raw.plot_psd(area_mode='range', tmax=10.0, picks=None, average=False)
# Why frequencies = 50, 340 are dashed ?
# Answer : The line noise frequency is
# also indicated with a dashed line (⋮) so probably frequencies after 340 are
# spotted as noise by mne and 50 was also spotted by mne.

# %%
# Low freq Pass ##################################################
##################################################################

raw.load_data()
raw.filter(l_freq=None, h_freq=h_freq)
raw_er.load_data()
raw_er.filter(l_freq=None, h_freq=h_freq)

# %%
# Savings ########################################################
##################################################################

raw.save(raw_maxfiltered_file)
raw_er.save(raw_er_maxfiltered_file)

# %%
