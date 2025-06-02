import os

import numpy as np
import pandas as pd

data_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(data_dir, 'processed')
results = {}

if __name__ == '__main__':
    train_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'train_data.pkl'))
    val_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'val_data.pkl'))
    test_data = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'test_data.pkl'))
    los_info = pd.read_pickle(os.path.join(processed_data_dir, 'split', 'los_info.pkl'))

    for split, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        patients = len(data)
        visits = sum(len(patient['x_ts']) for patient in data)
        avg_visits = visits / patients
        los = np.concatenate([patient['y_los'] for patient in data])
        los = los * los_info['los_std'] + los_info['los_mean']
        avg_los = np.mean(los)

        # Alive patients
        alive_patients = sum(patient['y_mortality'][0] == 0 for patient in data)
        alive_visits = sum(len(patient['x_ts']) for patient in data if patient['y_mortality'][0] == 0)
        alive_avg_visits = alive_visits / alive_patients
        alive_avg_los = np.mean(los[np.where(np.concatenate([patient['y_mortality'] for patient in data]) == 0)])

        # Dead patients
        dead_patients = sum(patient['y_mortality'][0] == 1 for patient in data)
        dead_visits = sum(len(patient['x_ts']) for patient in data if patient['y_mortality'][0] == 1)
        dead_avg_visits = dead_visits / dead_patients
        dead_avg_los = np.mean(los[np.where(np.concatenate([patient['y_mortality'] for patient in data]) == 1)])

        results[split] = {
            'patients': patients,
            'visits': visits,
            'avg_visits': avg_visits,
            'avg_los': avg_los,
            'alive_patients': alive_patients,
            'alive_visits': alive_visits,
            'alive_avg_visits': alive_avg_visits,
            'alive_avg_los': alive_avg_los,
            'dead_patients': dead_patients,
            'dead_visits': dead_visits,
            'dead_avg_visits': dead_avg_visits,
            'dead_avg_los': dead_avg_los
        }

    for split, result in results.items():
        print(f'{split} split:')
        print(f'\# Patients & {result["patients"]} & {result["alive_patients"]} & {result["dead_patients"]} \\\\')
        print(f'\# Total visits & {result["visits"]} & {result["alive_visits"]} & {result["dead_visits"]} \\\\')
        print(f'\# Avg. visits & {result["avg_visits"]:.1f} & {result["alive_avg_visits"]:.1f} & {result["dead_avg_visits"]:.1f} \\\\')
        print(f'Avg. LOS & {result["avg_los"]:.1f} & {result["alive_avg_los"]:.1f} & {result["dead_avg_los"]:.1f} \\\\')
        print()