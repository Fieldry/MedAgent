import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from pyehr.datasets.utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time

load_dotenv()
project_root = os.getenv("PROJECT_ROOT", ".")
data_dir = os.path.join(project_root, "my_datasets", "ehr", "obstetrics")
raw_data_dir = os.path.join(data_dir, "raw")
processed_data_dir = os.path.join(data_dir, "processed")
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome']
demographic_features = []
labtest_features = [
    'PT', 'APTT', 'TT', 'Fib', 'PAPTT', 'PTT', 'GLU', 'WBC', 'LY', 'NE', 'MO',
    'BAS', 'EOS', 'LY_pctn', 'NE_pctn', 'MO_pctn', 'BAS_pctn', 'EOS_pctn',
    'RBC', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'RDW_SD', 'PLT',
    'MPV', 'PCT', 'PDW', 'Cervical_length', 'Age'
]
require_impute_features = labtest_features
normalize_features = labtest_features

for kind in ['solo', 'multi']:
    labtest_df = pd.read_csv(os.path.join(raw_data_dir, f"labtest_{kind}.csv"))
    label_df = pd.read_csv(os.path.join(raw_data_dir, f"label_{kind}.csv"))

    label_df = label_df.rename(columns={'label:sptb': 'Outcome'})
    label_df["Outcome"] = label_df["Outcome"].apply(lambda x: 0 if x == '足月分娩' else 1)
    df = pd.merge(labtest_df, label_df[['PatientID', 'Outcome']], on='PatientID', how='left')
    df.dropna(subset=['Outcome'], inplace=True)
    df['Outcome'] = df['Outcome'].astype(int)

    df['RecordTime'] = pd.to_datetime(df['RecordTime'])

    formatted_parquet_path = os.path.join(processed_data_dir, f"obstetrics_{kind}_dataset_formatted.parquet")
    df.to_parquet(formatted_parquet_path)

    df = pd.read_parquet(formatted_parquet_path)

    df = df.sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)

    grouped = df.groupby('PatientID')

    patients = np.array(list(grouped.groups.keys()))

    patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

    train_val_patients, test_patients, train_val_outcomes, _ = train_test_split(
        patients, patients_outcome, test_size=300, random_state=SEED, stratify=patients_outcome
    )

    if len(train_val_patients) > 10000:
        _, train_val_patients, _, train_val_outcomes = train_test_split(
            train_val_patients, train_val_outcomes, test_size=10000, random_state=SEED, stratify=train_val_outcomes
        )

    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients, train_val_outcomes, test_size=1/8, random_state=SEED, stratify=train_val_outcomes
    )

    assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage between train and val sets"
    assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage between train and test sets"
    assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage between val and test sets"

    train_df = df[df['PatientID'].isin(train_patients)]
    val_df = df[df['PatientID'].isin(val_patients)]
    test_df = df[df['PatientID'].isin(test_patients)]

    train_missing_mask = export_missing_mask(train_df, demographic_features, labtest_features)
    val_missing_mask = export_missing_mask(val_df, demographic_features, labtest_features)
    test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features)

    train_record_time = export_record_time(train_df)
    val_record_time = export_record_time(val_df)
    test_record_time = export_record_time(test_df)

    _, train_raw_x, _, _ = forward_fill_pipeline(train_df, None, demographic_features, labtest_features, target_features, [])
    _, val_raw_x, _, _ = forward_fill_pipeline(val_df, None, demographic_features, labtest_features, target_features, [])
    _, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [])

    # For dl setting, export data on train/val/test set:
    train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

    train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
    test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

    train_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_sptb': [y[0] for y in y_item], # sptb: Spontaneous Preterm Birth
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(train_pid, train_x, train_raw_x, train_record_time, train_missing_mask, train_y)]
    val_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_sptb': [y[0] for y in y_item],
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(val_pid, val_x, val_raw_x, val_record_time, val_missing_mask, val_y)]
    test_data = [{
        'id': id_item,
        'x_ts': x_item,
        'x_llm_ts': x_llm_item,
        'record_time': record_time_item,
        'missing_mask': missing_mask_item,
        'y_sptb': [y[0] for y in y_item],
    } for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(test_pid, test_x, test_raw_x, test_record_time, test_missing_mask, test_y)]

    test_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in test_patients])

    sub_train_val_patients, sub_test_patients, sub_train_val_outcomes, _ = train_test_split(
        test_patients, test_patients_outcome, test_size=200, random_state=SEED, stratify=test_patients_outcome
    )
    sub_train_patients, sub_val_patients, _, _ = train_test_split(
        sub_train_val_patients, sub_train_val_outcomes, test_size=50, random_state=SEED, stratify=sub_train_val_outcomes
    )

    sub_train_data = [item for item in test_data if item['id'] in sub_train_patients]
    sub_val_data = [item for item in test_data if item['id'] in sub_val_patients]
    sub_test_data = [item for item in test_data if item['id'] in sub_test_patients]

    save_dir = os.path.join(processed_data_dir, kind)
    os.makedirs(save_dir, exist_ok=True)

    pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
    pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
    pd.to_pickle(test_data, os.path.join(save_dir, "fusion_data.pkl"))
    pd.to_pickle(sub_train_data, os.path.join(save_dir, "fusion_train_data.pkl"))
    pd.to_pickle(sub_val_data, os.path.join(save_dir, "fusion_val_data.pkl"))
    pd.to_pickle(sub_test_data, os.path.join(save_dir, "test_data.pkl"))

    print("\n--- Final Data Sizes ---")
    print("Train data size:", len(train_data))
    print("Validation data size:", len(val_data))
    print("Fusion data (full test set) size:", len(test_data))
    print("Fusion train data size:", len(sub_train_data))
    print("Fusion val data size:", len(sub_val_data))
    print("Test data size:", len(sub_test_data))

    if los_info is not None:
        pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))

    pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'negative_stats.pkl'))
    pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'positive_stats.pkl'))

    pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))