""" Coregistration. IN this file you should:
    - Open the gui
    - Verify the regitration.
    - Save the registration from the gui manually to the file data/trans in the file: trans

"""
# %%
import mne
from params import subjects_dir, subject, sample_data_raw_file


mne.gui.coregistration(tabbed=None, split=True,
                       inst=sample_data_raw_file,
                       subject=subject, subjects_dir=subjects_dir)


# %%
