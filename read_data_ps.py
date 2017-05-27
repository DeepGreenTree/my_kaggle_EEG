"""
Cache the train and test data for the EEG Predict Seizure (PS) competition.

This is a temporary measure to conform to the RCNN code for the EEG Grasp and
Lift (GAL) competition. Unlike GAL, we cache the data per subject because we
create a separate model for each subject.

The data is saved as an array of the form [data, label]

For train, the labels are those supplied. For test, the labels are set to zero.
"""
import os

import numpy as np
import pandas as pd

# csvdir = 'data/train'
csvdir = '../kaggle-eeg-ps/train_'
n_subs = 3
n_series = 8
n_channels = 32

data = []
label = []

#########
# Train #
#########

# For each subject
for sub in np.arange(n_subs):
    sub_data = []
    sub_label = []

    # For each series
    for series in np.arange(n_series):

        # Read this data
        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_data.csv'
        series_data = pd.read_csv(os.path.join(csvdir, csv))

        # Add the data (without the ids) to our collection
        ch_names = list(series_data.columns[1:])
        series_data = np.array(series_data[ch_names], 'float32')
        sub_data.append(series_data)

        # Read the corresponding events
        csv = 'subj' + str(sub + 1) + '_series' + str(series + 1) + '_events.csv'
        series_label = pd.read_csv(os.path.join(csvdir, csv))

        # Add the events (without the ids) to our collection
        ch_names = list(series_label.columns[1:])
        series_label = np.array(series_label[ch_names], 'float32')
        sub_label.append(series_label)

    data.append(sub_data)
    label.append(sub_label)

# Save
np.save('eeg_train.npy', [data, label])

########
# Test #
########

csvdir = 'data/test'
n_subs = 12
n_series = 2
n_channels = 32

data = []
label = []

# For each subject
for sub in np.arange(n_subs):
    sub_data = []
    sub_label = []

    # For each series
    for series in np.arange(9, 9 + n_series):

        # Read the data
        csv = 'subj' + str(sub + 1) + '_series' + str(series) + '_data.csv'
        series_data = pd.read_csv(os.path.join(csvdir, csv))

        # Add the data (without the ids) to our collection
        ch_names = list(series_data.columns[1:])
        series_data = np.array(series_data[ch_names], 'float32')
        sub_data.append(series_data)

        # Add placeholder labels to our collection
        series_label = np.zeros([series_data.shape[0], 6])
        sub_label.append(series_label)

    data.append(sub_data)
    label.append(sub_label)

np.save('eeg_test.npy', [data, label])
