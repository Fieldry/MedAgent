import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from dotenv import load_dotenv

from pyehr.datasets.utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time

load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
data_dir = os.path.join(project_root, "my_datasets", "ehr", "esrd")
raw_data_dir = os.path.join(data_dir, "raw")
processed_data_dir = os.path.join(data_dir, "processed")
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

# Record feature names
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome']
demographic_features = []
labtest_features = ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose', 'Appetite', 'Weight', 'SBP', 'DBP']
require_impute_features = labtest_features
normalize_features = labtest_features

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'esrd_dataset_formatted.parquet'))

# Ensure the data is sorted by PatientID and RecordTime
df = df.sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)

num_folds = 10

# Group the dataframe by `PatientID`
grouped = df.groupby('PatientID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

for fold, (train_val_index, test_index) in enumerate(kf.split(patients, patients_outcome)):
    train_val_patients = patients[train_val_index]
    test_patients = patients[test_index]

    train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
    train_patients, val_patients = train_test_split(train_val_patients, test_size=1/(num_folds - 1), random_state=SEED, stratify=train_val_patients_outcome)

    train_df = df[df['PatientID'].isin(train_patients)]
    val_df = df[df['PatientID'].isin(val_patients)]
    test_df = df[df['PatientID'].isin(test_patients)]

    # Export the missing mask
    train_missing_mask = export_missing_mask(train_df, demographic_features, labtest_features)
    val_missing_mask = export_missing_mask(val_df, demographic_features, labtest_features)
    test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features)

    # Export the record time
    train_record_time = export_record_time(train_df)
    val_record_time = export_record_time(val_df)
    test_record_time = export_record_time(test_df)

    # Export the raw data
    _, train_raw_x, _, _ = forward_fill_pipeline(train_df, None, demographic_features, labtest_features, target_features, [])
    _, val_raw_x, _, _ = forward_fill_pipeline(val_df, None, demographic_features, labtest_features, target_features, [])
    _, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [])

    # Normalize the train, val, test data
    train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

    # Forward Imputation after grouped by PatientID
    train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

    # Convert the data to the required format
    train_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_mortality': [y[0] for y in y_item],
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(train_pid, train_x, train_raw_x, train_record_time, train_missing_mask, train_y)]
    val_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_mortality': [y[0] for y in y_item],
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(val_pid, val_x, val_raw_x, val_record_time, val_missing_mask, val_y)]
    test_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_mortality': [y[0] for y in y_item],
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(test_pid, test_x, test_raw_x, test_record_time, test_missing_mask, test_y)]


    # Create the directory to save the processed data
    save_dir = os.path.join(processed_data_dir, f"fold_{fold}")
    os.makedirs(save_dir, exist_ok=True)

    # Save the data to pickle files
    pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
    pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
    pd.to_pickle(test_data, os.path.join(save_dir, "test_data.pkl"))

    # Print the sizes of the datasets
    print("Train data size:", len(train_data))
    print("Validation data size:", len(val_data))
    print("Test data size:", len(test_data))
