"""
ICA to remove eye-blinks and cardiac artifacts
"""
# %%
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs)
import mne
from params import (raw_maxfiltered_file, sample_data_raw_file,
                    h_freq, FIND_EVENTS_KWARGS, event_id)


raw = mne.io.read_raw_fif(raw_maxfiltered_file)

##############################################################################
# Visualisation of the artifacts ()
# ----------------------


# %%
# Let’s begin by visualizing the artifacts that we want to repair.
# In this dataset they are big enough to see easily in the raw data:
# pick some channels that clearly show heartbeats and blinks
regexp = r'(MEG[12][45][123]1)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
         show_scrollbars=False)
# MEG1411, MEG1421, MEG1431 present clearly cardiac artifacts.


# %%
# We can get a summary of how the
# ocular artifact manifests across each channel type
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()


##############################################################################
# Isolation of the artifacts with ICA
# raw_base > 1hz > new epochs > ICA
# and not epoch > 1hz > ICA to avoid artifacts.
# ----------------------

# %%
raw_base = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
raw_base.filter(l_freq=1, h_freq=h_freq)
events = mne.find_events(raw_base, **FIND_EVENTS_KWARGS)
epochs = mne.Epochs(raw_base, events, event_id=event_id,
                    on_missing='ignore')

epochs = epochs.copy()
ica = ICA(n_components=15, random_state=1)
ica.fit(epochs)
# BUG: that's not really cardiac artifacts.


# %%
ica.plot_components()

# %%
# blinks
ica.plot_overlay(raw_base, exclude=[0], picks='mag')
# heartbeats
ica.plot_overlay(raw_base, exclude=[1], picks='mag')

# %%
ica.plot_properties(raw_base, picks=[0, 1])


# %%
##############################################################################
# Selecting ICA components manually¶
# Once we’re certain which components we want to exclude,
# We can specify that manually by setting the ica.exclude
# ----------------------

ica.exclude = [0, 1]  # indices chosen based on various plots above
reconst_raw = raw_base.copy()
ica.apply(reconst_raw)

raw_base.plot(order=artifact_picks, n_channels=len(artifact_picks))
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
# %%
