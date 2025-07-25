import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from pyehr.datasets.utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time, export_note

load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
data_dir = os.path.join(project_root, "my_datasets", "ehr", "mimic-iv")
raw_data_dir = os.path.join(data_dir, "raw")
processed_data_dir = os.path.join(data_dir, "processed")
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

# Record feature names
basic_records = ['RecordID', 'PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
note_features = ['Text']
demographic_features = ['Sex', 'Age']
labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# Stratified split dataset into train, validation and test sets
# For ml/dl models: include Imputation & Normalization & Outlier Filtering steps
# For all settings, randomly select 300 patients for test set
# Then randomly select 10000 in the rest used for training and validation (7/8 training, 1/8 validation)

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic-iv-timeseries-note.parquet'))
df = df[basic_records + target_features + note_features + demographic_features + labtest_features]

# For ml/dl models, convert categorical features to one-hot encoding
one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
columns = df.columns.to_list()
column_start_idx = columns.index(categorical_labtest_features[0])
column_end_idx = columns.index(categorical_labtest_features[-1])
df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)

# Update the categorical lab test features
ehr_categorical_labtest_features = one_hot.columns.to_list()
ehr_labtest_features = ehr_categorical_labtest_features + numerical_labtest_features
require_impute_features = ehr_labtest_features

# Group the dataframe by patient ID
grouped = df.groupby('RecordID')

# Randomly select 300 patients for the test set
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])
train_val_patients, test_patients, train_val_patients_outcome, _ = train_test_split(patients, patients_outcome, test_size=300, random_state=SEED, stratify=patients_outcome)

# Randomly select 10000 patients for the train/val set
_, train_val_patients, _, train_val_patients_outcome = train_test_split(train_val_patients, train_val_patients_outcome, test_size=10000, random_state=SEED, stratify=train_val_patients_outcome)

# Split the train/val set into train and val sets, 7/8 for train and 1/8 for val
train_patients, val_patients = train_test_split(train_val_patients, test_size=1/8, random_state=SEED, stratify=train_val_patients_outcome)

# Print the sizes of the datasets
print("Train patients size:", len(train_patients))
print("Validation patients size:", len(val_patients))
print("Test patients size:", len(test_patients))

# Assert there is no data leakage
assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage between train and val sets"
assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage between train and test sets"
assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage between val and test sets"

# Create train, val dataframes
train_df = df[df['RecordID'].isin(train_patients)]
val_df = df[df['RecordID'].isin(val_patients)]
test_df = df[df['RecordID'].isin(test_patients)]

# Export the missing mask
train_missing_mask = export_missing_mask(train_df, demographic_features, ehr_labtest_features, id_column='RecordID')
val_missing_mask = export_missing_mask(val_df, demographic_features, ehr_labtest_features, id_column='RecordID')
test_missing_mask = export_missing_mask(test_df, demographic_features, ehr_labtest_features, id_column='RecordID')

# Export the record time
train_record_time = export_record_time(train_df, id_column='RecordID')
val_record_time = export_record_time(val_df, id_column='RecordID')
test_record_time = export_record_time(test_df, id_column='RecordID')

# Export the raw data
_, train_raw_x, _, _ = forward_fill_pipeline(train_df, None, demographic_features, ehr_labtest_features, target_features, [], id_column='RecordID')
_, val_raw_x, _, _ = forward_fill_pipeline(val_df, None, demographic_features, ehr_labtest_features, target_features, [], id_column='RecordID')
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, ehr_labtest_features, target_features, [], id_column='RecordID')

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features, id_column="RecordID")

# Forward Imputation after grouped by RecordID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")

# Export the note
train_note = export_note(train_df, id_column='RecordID')
val_note = export_note(val_df, id_column='RecordID')
test_note = export_note(test_df, id_column='RecordID')

# Create the directory to save the processed data
save_dir = os.path.join(processed_data_dir, 'split')
os.makedirs(save_dir, exist_ok=True)

# Convert the data to the required format
train_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, x_llm_item, note_item, record_time_item, missing_mask_item, y_item in zip(train_pid, train_x, train_raw_x, train_note, train_record_time, train_missing_mask, train_y)]
val_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, x_llm_item, note_item, record_time_item, missing_mask_item, y_item in zip(val_pid, val_x, val_raw_x, val_note, val_record_time, val_missing_mask, val_y)]
test_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, x_llm_item, note_item, record_time_item, missing_mask_item, y_item in zip(test_pid, test_x, test_raw_x, test_note, test_record_time, test_missing_mask, test_y)]

# Split test data into 3 parts: 150 patients for train, 50 patients for val, 100 patients for test
test_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in test_patients])
sub_train_val_patients, sub_test_patients = train_test_split(test_patients, test_size=200, random_state=SEED, stratify=test_patients_outcome)
sub_train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in sub_train_val_patients])
sub_train_patients, sub_val_patients = train_test_split(sub_train_val_patients, test_size=1/2, random_state=SEED, stratify=sub_train_val_patients_outcome)

sub_train_data = [item for item in test_data if item['id'] in sub_train_patients]
sub_val_data = [item for item in test_data if item['id'] in sub_val_patients]
sub_test_data = [item for item in test_data if item['id'] in sub_test_patients]

# Save the data to pickle files
pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
pd.to_pickle(test_data, os.path.join(save_dir, "fusion_data.pkl"))
pd.to_pickle(sub_train_data, os.path.join(save_dir, "fusion_train_data.pkl"))
pd.to_pickle(sub_val_data, os.path.join(save_dir, "fusion_val_data.pkl"))
pd.to_pickle(sub_test_data, os.path.join(save_dir, "test_data.pkl"))

# Print the sizes of the datasets
print("Train data size:", len(train_data))
print("Validation data size:", len(val_data))
print("Fusion data size:", len(test_data))
print("Fusion train data size:", len(sub_train_data))
print("Fusion val data size:", len(sub_val_data))
print("Test data size:", len(sub_test_data))

# Export LOS statistics (calculated from the train set)
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))

# Export the labtest feature names
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))
pd.to_pickle(ehr_labtest_features, os.path.join(save_dir, "ehr_labtest_features.pkl"))

# Export the basic info
pd.to_pickle({
    key: grouped.get_group(key)[['Sex', 'Age']].drop_duplicates().to_dict('records')[0]
    for key in grouped.groups.keys()
}, os.path.join(save_dir, 'basic.pkl'))

# Export the survival and death statistics
pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))