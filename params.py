"""
Parameters :
    - file paths
    - meta parameters
    - Event id

"""

# FILES  #########################################################
##################################################################

import os

subject = 'CC110033'
# subject = 'CC110037'

sample_data_raw_file = os.path.join("data",
                                    subject,
                                    'task',
                                    'task_raw.fif')

sample_data_raw_file_er = os.path.join(
    "data",
    "emptyroom",
    subject,
    'emptyroom_' + subject + '.fif')


fine_cal_file = os.path.join(
    "data",
    "welcome_duties_sss_cal.dat")

# Is there a crosstalk file anywhere ?
# crosstalk_file = os.path.join(sample_data_folder, 'SSS', 'ct_sparse_mgh.fif')

raw_maxfiltered_file = os.path.join(
    "data",
    "processing",
    subject,
    'raw_maxwell_filtered.fif')

raw_er_maxfiltered_file = os.path.join(
    "data",
    "processing",
    subject,
    "raw_er_maxwell_filtered.fif")

epochs_file = os.path.join(
    "data",
    "processing",
    subject,
    subject + "-epo.fif")

evoked_file = os.path.join(
    "data",
    "processing",
    subject,
    subject + "-ave.fif")

noise_cov_baseline_file = os.path.join(
    "data",
    "processing",
    subject,
    subject + "-cov.fif")

# Coregistration
subjects_dir = os.path.join("data",
                            "freesurfer")

trans_dir = os.path.join("data",
                         "trans",
                         "sub-" + subject + "-trans.fif")

# Forward and Inverse
bem_dir = os.path.join("data",
                       "freesurfer",
                       subject,
                       "bem",
                       subject+"-meg-bem.fif")


# Params  ########################################################
##################################################################

h_freq = 40

# Event  #########################################################
##################################################################

event_id = {
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

AUTOREJECT = False  # Warning : this takes a lot of RAM and lot of time...

FIND_EVENTS_KWARGS = {
    "min_duration": 0.001,
    "shortest_event": 2
}
