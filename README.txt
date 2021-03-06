Author: Charbel-Raphael Segerie <charbel-raphael.segerie@inria.fr>

Tasks:

maxfilter.py:

- Preprocess the data using maxfilter. Calibration files for maxfilter
can be found in this parietal repo ("sss_cal.dat").
You can plot the PSD to have a first look at the signals and see
the effects of the filtering.
This tutorial might be useful for understanding this task.
Make sure to look at the filtered data to define bad channels. Definition
 of bad channels should be done manually by visual inspection
(looking up the original CAMCAN maxfilter log is not permitted). You can
eventually have a look at this tutorial for some help.

- Bandpass filter the data in a reasonable range (look what evoked studies
usually do), see. Don't forget to clean the power line frequencies (see).


epoch.py:

- Construct epochs from MEG data, see. Event codes can be found in
/storage/store/data/camcan/camcan47/cc700/meg/pipeline/release004/data_info/trigger_codes.txt.
 Set manually adequate reject parameter in Epochs or use autoreject local
 to clean the data from remaining artifacts.

- Compute evoked responses for the visual and auditory events.
When does the signal peaks? Does the topographies look dipolar at the peak latencies?


noise_cov.py:

- Compute the noise covariance from the baseline segments with optimal shrinkage
 and document the quality of the resulting spatial - whitening, see.

- Also compute the noise covariance from the empty room recordings
and compare it with the one obtained from the task.
Note that the empty room needs to undergo the same signal treatment
as the actual MEG recording in order have an appropriate covariance matrix.
Empty room recordings can be found in
/storage/store/data/camcan/camcan47/cc700/meg/pipeline/release004/emptyroom/


coreg.py:

- Do the co-registration to obtain the head-to-MRI transform using the MNE
coregistration GUI.
For some subjects, existing transformation files can be found in drago
here /storage/store/data/camcan-mne/trans/ and MRIs can be found in /storage/store/data/camcan-mne/freesurfer/.
Plot transformations to compare the results.

inv.py:

- Compute the dSPM inverse solution on the cortical surface
for both covariance matrices. You will need to make a forward solution first
(you can create a forward solution after setting up a source space,
making a bem model and bem solution).
All the files you need are available in the folder subjects_dir in /freesurfer.
 Compare the results.
- Does the peak location correspond to known anatomical locations of the primary
auditory and visual cortices?

ica.py

Use ICA to remove eye-blinks and cardiac artifacts, see the introduction to ICA,
the artifact correction with ICA,
the comparison of different ICA algorithms.
You can install this package first if you want to try the more efficient Picard
 algorithm.
