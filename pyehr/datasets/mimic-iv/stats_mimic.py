import os

import pandas as pd

data_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(data_dir, 'processed')
results = {}

if __name__ == '__main__':
    train_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'train_data.pkl'))
    val_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'val_data.pkl'))
    test_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'test_data.pkl'))


    for split, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        patients = len(data)
        visits = sum(len(patient['x_ts']) for patient in data)
        avg_visits = visits / patients

        alive_patients = sum(patient['y_mortality'][0] == 0 for patient in data)
        alive_visits = sum(len(patient['x_ts']) for patient in data if patient['y_mortality'][0] == 0)
        alive_avg_visits = alive_visits / alive_patients

        dead_patients = sum(patient['y_mortality'][0] == 1 for patient in data)
        dead_visits = sum(len(patient['x_ts']) for patient in data if patient['y_mortality'][0] == 1)
        dead_avg_visits = dead_visits / dead_patients

        readmission_patients = sum(patient['y_readmission'][0] == 1 for patient in data)
        readmission_visits = sum(len(patient['x_ts']) for patient in data if patient['y_readmission'][0] == 1)
        readmission_avg_visits = readmission_visits / readmission_patients

        no_readmission_patients = sum(patient['y_readmission'][0] == 0 for patient in data)
        no_readmission_visits = sum(len(patient['x_ts']) for patient in data if patient['y_readmission'][0] == 0)
        no_readmission_avg_visits = no_readmission_visits / no_readmission_patients

        results[split] = {
            'patients': patients,
            'visits': visits,
            'avg_visits': avg_visits,
            'alive_patients': alive_patients,
            'alive_visits': alive_visits,
            'alive_avg_visits': alive_avg_visits,
            'dead_patients': dead_patients,
            'dead_visits': dead_visits,
            'dead_avg_visits': dead_avg_visits,
            'readmission_patients': readmission_patients,
            'readmission_visits': readmission_visits,
            'readmission_avg_visits': readmission_avg_visits,
            'no_readmission_patients': no_readmission_patients,
            'no_readmission_visits': no_readmission_visits,
            'no_readmission_avg_visits': no_readmission_avg_visits
        }

    for split, result in results.items():
        print(f'{split} split:')
        print(f'\# Patients & {result["patients"]} & {result["alive_patients"]} & {result["dead_patients"]} & {result["readmission_patients"]} & {result["no_readmission_patients"]} \\\\')
        print(f'\# Total visits & {result["visits"]} & {result["alive_visits"]} & {result["dead_visits"]} & {result["readmission_visits"]} & {result["no_readmission_visits"]} \\\\')
        print(f'\# Avg. visits & {result["avg_visits"]:.1f} & {result["alive_avg_visits"]:.1f} & {result["dead_avg_visits"]:.1f} & {result["readmission_avg_visits"]:.1f} & {result["no_readmission_avg_visits"]:.1f} \\\\')
        print()