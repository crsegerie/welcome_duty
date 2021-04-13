"""
ICA to remove eye-blinks and cardiac artifacts
"""

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs)
import mne

from params import raw_maxfiltered_file


raw = mne.io.read_raw_fif(raw_maxfiltered_file)


# %%
# We can get a summary of how the
# ocular artifact manifests across each channel type
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()

# %%

filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=None)
ica.fit(filt_raw)

# %%
# BUG in the data
