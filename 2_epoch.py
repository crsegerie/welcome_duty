"""
Construct epoch from maxfiltered data:
    - create epoch.
    - auto reject epochs
    - create Evoked data
    - save Epochs, Evoked
    - Topographic map

"""
# %%
# Imports
import numpy as np

from autoreject import get_rejection_threshold
from autoreject import AutoReject
import mne

from params import (event_name_to_id_mapping,
                    raw_maxfiltered_file, raw_er_maxfiltered_file,
                    epochs_file, evoked_file, AUTOREJECT,
                    FIND_EVENTS_KWARGS)

raw = mne.io.read_raw_fif(raw_maxfiltered_file)
raw_er = mne.io.read_raw_fif(raw_er_maxfiltered_file)

# %%
# Construct epochs from MEG data
events = mne.find_events(raw, **FIND_EVENTS_KWARGS)
epochs = mne.Epochs(raw, events, event_id=event_name_to_id_mapping,
                    on_missing='ignore', )

for (e, i) in event_name_to_id_mapping.items():
    a = (events[:, -1] == i).sum()
    print(f"event {e} is present {a} times")

# %% use autoreject local to clean the data from remaining artifacts
if AUTOREJECT:
    ar = AutoReject()
    epochs.load_data()
    epochs_clean = ar.fit_transform(epochs)
else:
    epochs_clean = epochs


# %%
# Is this related with the bonferroni correction ?
reject = get_rejection_threshold(epochs)
print(reject)

# %%
evoked = epochs_clean['audiovis/1200Hz'].average()
evoked.plot()

# %%
epochs_clean.save(epochs_file, overwrite=True)
evoked.save(evoked_file)


# %% [markdown]
# ### When does the signal peaks? Does the topographies look dipolar
# at the peak
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
# > Yes
