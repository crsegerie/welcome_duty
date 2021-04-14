""" Coregistration. IN this file you should:
    - Open the gui
    - Verify the regitration.
    - Save the registration from the gui manually to the file data/trans

"""


import mne
from params import subjects_dir, trans_dir, subject, sample_data_raw_file


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


# mne.viz.plot_alignment(info=None, trans=trans_dir, subject=subject,
#                        subjects_dir=subjects_dir, surfaces='auto',
#                        coord_frame='head', meg=None,
#                        eeg='original', fwd=None, dig=False,
#                        ecog=True, src=None,
#                        mri_fiducials=False, bem=None, seeg=True,
#                        fnirs=True, show_axes=False, fig=None,
#                        interaction='trackball', verbose=None)
