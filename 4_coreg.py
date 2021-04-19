""" Coregistration. IN this file you should:
    - Open the gui
    - Verify the regitration.
    - Save the registration from the gui manually to the file data/trans in the file: trans

"""
# %%
import mne
from params import (subjects_dir, subject, sample_data_raw_file,
                    trans_dir, sample_data_raw_file)

print("Please save the trans file here:", trans_dir)
mne.gui.coregistration(tabbed=None, split=True,
                       inst=sample_data_raw_file,
                       subject=subject, subjects_dir=subjects_dir)


# # %%
# info = mne.io.read_info(sample_data_raw_file)
# # Here we look at the dense head, which isn't used for BEM computations but
# # is useful for coregistration.
# mne.viz.plot_alignment(info, trans_dir, subject=subject, dig=True,
#                        meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
#                        surfaces='head-dense')
