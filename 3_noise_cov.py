"""
This file:
    - Compute noise cov from empty room
    - Compute the noise covariance from the baseline segments
    - Plot the two covariances
    - Saves the baseline covariance

"""
# %%
import mne

from params import (raw_maxfiltered_file, raw_er_maxfiltered_file,
                    noise_cov_baseline_file,
                    FIND_EVENTS_KWARGS)

raw = mne.io.read_raw_fif(raw_maxfiltered_file)
raw_er = mne.io.read_raw_fif(raw_er_maxfiltered_file)
events = mne.find_events(raw, **FIND_EVENTS_KWARGS)


# %%
# Compute the noise from empty room
raw_er.info['bads'] = [
    bb for bb in raw.info['bads'] if 'EEG' not in bb]
raw_er.add_proj(
    [pp.copy() for pp in raw.info['projs'] if 'EEG' not in pp['desc']])

noise_cov_er = mne.compute_raw_covariance(
    raw_er, tmin=0, tmax=None)

# %%
# We also use the pre-stimulus baseline to estimate the noise covariance
# epochs = mne.Epochs(raw, events, event_id=event_id,
#                         on_missing='ignore')

ante_epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                         # we'll decimate for speed
                         baseline=(-0.2, 0.0), decim=3,
                         verbose='error')  # and ignore the warning about aliasing

noise_cov_baseline = mne.compute_covariance(ante_epochs, tmax=0)

# %%
noise_cov_er.plot(raw_er.info, proj=True)
noise_cov_baseline.plot(ante_epochs.info, proj=True)
noise_cov_baseline.save(noise_cov_baseline_file)
# BUG: ON the second individual the er grad covariance is Null ?

# %%
