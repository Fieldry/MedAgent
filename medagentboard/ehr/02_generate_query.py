import os
import argparse
import json
from collections import OrderedDict
from typing import List, Tuple, Any, Dict

import pandas as pd
import numpy as np

from medagentboard.utils.datasets_info import medical_name, medical_unit, medical_standard, original_disease, disease_english
from medagentboard.utils.generate_obste_note import generate_obste_note


def generate_prompt(
    dataset: str,
    task: str,
    models: List[str],
    basic_data: Dict,
    ys: List[float],
    important_features: List[List[Tuple[str, Dict]]],
    survival_stats: Dict,
    dead_stats: Dict,
) -> Tuple[str, str]:
    if dataset == 'esrd':
        gender = "male" if basic_data["Gender"] == 1 else "female"
        age = basic_data["Age"]
        if " " in basic_data["Origin_disease"]:
            ori_disease = basic_data["Origin_disease"].split(" ")[0]
            ori_disease = original_disease[ori_disease]
        else:
            ori_disease = original_disease[basic_data["Origin_disease"]]
        basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
        basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
        basic_context = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"
    elif dataset == 'cdsl':
        gender = "male" if basic_data["Sex"] == 1 else "female"
        age = basic_data["Age"]
        basic_context = f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"
    elif dataset == 'obstetrics':
        basic_context = generate_obste_note(basic_data)
    else:  # [mimic-iii, mimic-iv]
        gender = "male" if basic_data["Sex"] == 1 else "female"
        age = basic_data["Age"]
        basic_context = f"This {gender} patient, aged {age}, is an patient in intensive care unit (ICU).\n"

    last_visit_all_context = f"We have {len(models)} models {', '.join(models)} to predict the {task} risk and estimate the feature importance weight for the patient in the last visit:\n"
    last_visit_contexts = []
    for model, y, important_features_item in zip(models, ys, important_features):
        last_visit_context = f"We have model {model} to predict the {task} risk and estimate the feature importance weight for the patient in the last visit:\n"
        last_visit = f"The {task} prediction risk for the patient from {model} model is {round(float(y), 2)} out of 1.0, which means the patient is at {get_death_desc(float(y))} of death risk. Our model especially pays great attention to the following features:\n"
        for item in important_features_item:
            key, value, attention = item
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            key_name = medical_name[key] if key in medical_name else key
            key_unit = ' ' + medical_unit[key] if key in medical_unit else ''
            last_visit += f'{key_name}: with '
            last_visit += f'importance weight of {round(float(attention), 3)} out of 1.0. '
            last_visit += f'The feature value is {round(value, 2)}{key_unit}, which is {get_mean_desc(value, survival_mean)} than the average value of negative samples ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value, dead_mean)} than the average value of positive samples ({round(dead_mean, 2)}{key_unit}).\n'
        last_visit_context += last_visit
        last_visit_contexts.append(last_visit_context)
        last_visit_all_context += last_visit

    last_visit_contexts.append(last_visit_all_context)
    return basic_context, last_visit_contexts


def format_input_ehr(
    patient: List[List[float]],
    dataset: str,
    features: List[str],
    mask: List[List[int]],
    record_time: List[str],
    demo_dim: int=2
) -> str:
    """
    Format patient data for LLM input.

    Args:
        patient: List of patient visits
        dataset: Dataset name ('mimic-iv' or 'tjh')
        features: List of feature names
        mask: Missing value masks

    Returns:
        Formatted string with patient details
    """
    assert len(patient[0]) == demo_dim + len(features), f"Patient data length mismatch: expected {demo_dim + len(features)}, got {len(patient[0])}"
    assert len(mask[0]) == demo_dim + len(features), f"Mask length mismatch: expected {demo_dim + len(features)}, got {len(mask[0])}"
    assert len(record_time) == len(patient)

    patient = np.array(patient)[:, demo_dim:].tolist()
    mask = np.array(mask)[:, demo_dim:].tolist()

    # Define some categorical features with their possible values
    categorical_features_dict = {
        "Glascow coma scale eye opening": {
            1: "No Response",
            2: "To Pain",
            3: "To Speech",
            4: "Spontaneously",
        },
        "Glascow coma scale motor response": {
            1: "No Response",
            2: "Abnormal Extension",
            3: "Abnormal Flexion",
            4: "Flex-withdraws",
            5: "Localizes Pain",
            6: "Obeys Commands",
        },
        "Glascow coma scale verbal response": {
            1: "No Response",
            2: "Incomprehensible sounds",
            3: "Inappropriate Words",
            4: "Confused",
            5: "Oriented",
        },
    }

    grouped_features = OrderedDict()
    for i, feature_name in enumerate(features):
        if '->' in feature_name:
            base_name, value_str = feature_name.split('->', 1)
            value = float(value_str)
            if base_name not in grouped_features:
                grouped_features[base_name] = {
                    'type': 'categorical',
                    'components': []
                }
            grouped_features[base_name]['components'].append({'index': i, 'value': value})
        else:
            grouped_features[feature_name] = {
                'type': 'continuous',
                'index': i
            }

    feature_values = {key: [] for key in grouped_features.keys()}

    for visit_idx in range(len(patient)):
        visit_data = patient[visit_idx]
        visit_mask = mask[visit_idx]

        for base_name, info in grouped_features.items():
            final_value = 'NaN'

            if info['type'] == 'categorical':
                is_missing = True
                for component in info['components']:
                    idx = component['index']
                    if visit_mask[idx] == 0:
                        is_missing = False
                        if visit_data[idx] == 1.0:
                            category_value = component['value']
                            if base_name in categorical_features_dict and categorical_features_dict[base_name]:
                                final_value = categorical_features_dict[base_name].get(category_value, str(category_value))
                            else:
                                final_value = str(int(category_value))
                            break
                if is_missing:
                    final_value = 'NaN'

            elif info['type'] == 'continuous':
                idx = info['index']
                if visit_mask[idx] == 0:
                    final_value = f"{visit_data[idx]:.2f}"

            feature_values[base_name].append(final_value)

    detail = ''
    if record_time is not None:
        assert len(patient) == len(record_time), "The length of patient and record_time should be the same."
        detail += "The patient's EHR data is recorded at the following time points:\n"
        detail += ", ".join(record_time) + ".\n"

    for feature_name in grouped_features.keys():
        values_str = ", ".join(feature_values[feature_name])
        detail += f'- {feature_name}: [{values_str}]\n'

    return detail.strip() + '\n'


def load_dataset(root_path: str, dataset: str, task: str, split: str="split") -> Tuple[List, List, List, List, List, List, List, Any]:
    """
    Load dataset based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of dataset components
    """
    dataset_path = os.path.join(root_path, f'{dataset}/processed/{split}')
    data = pd.read_pickle(os.path.join(dataset_path, 'fusion_data.pkl'))
    ids = [item['id'] for item in data]
    xs = [item['x_ts'] for item in data]
    x_llm_ts = [item['x_llm_ts'] for item in data]
    ys = [item[f'y_{task}'] for item in data]
    missing_masks = [item['missing_mask'] for item in data]
    record_times = [item['record_time'] for item in data]

    if dataset == 'mimic-iv':
        labtest_features = pd.read_pickle(os.path.join(dataset_path, 'ehr_labtest_features.pkl'))
        x_note = [item['x_note'] for item in data]
    else:
        labtest_features = pd.read_pickle(os.path.join(dataset_path, 'labtest_features.pkl'))
        x_note = None

    if dataset == 'obstetrics':
        positive_stats = pd.read_pickle(os.path.join(dataset_path, 'positive_stats.pkl'))
        negative_stats = pd.read_pickle(os.path.join(dataset_path, 'negative_stats.pkl'))
        basic_info = pd.read_csv(os.path.join(dataset_path, f'demo_{split}.csv'))
        basic_info = {item['PatientID']: item for item in basic_info.to_dict(orient='records')}
    else:
        positive_stats = pd.read_pickle(os.path.join(dataset_path, 'dead.pkl'))
        negative_stats = pd.read_pickle(os.path.join(dataset_path, 'survival.pkl'))
        basic_info = pd.read_pickle(os.path.join(dataset_path, 'basic.pkl'))

    if task == 'los':
        try:
            los_info = pd.read_pickle(os.path.join(dataset_path, 'los_info.pkl'))
        except FileNotFoundError:
            raise FileNotFoundError(f"LOS info file not found in {dataset_path}.")
    else:
        los_info = None

    return ids, xs, x_llm_ts, x_note, ys, missing_masks, record_times, labtest_features, los_info, basic_info, negative_stats, positive_stats


def load_training_results(root_path: str, dataset: str, task: str, model: str) -> Tuple[List, List]:
    results_path = os.path.join(root_path, f'{dataset}/{task}/{model}')
    results = pd.read_pickle(os.path.join(results_path, 'outputs.pkl'))
    preds = results['preds']
    attn = results['attns']
    return preds, attn


def process_important_features(values: List[float], attns: List[float], features: List[str], demo_dim: int=2) -> List[Tuple[str, Dict]]:
    assert len(values) == demo_dim + len(features), f"Values length: {len(values)}, demo_dim: {demo_dim}, features: {len(features)}"
    assert len(attns) == len(features), f"Attns length: {len(attns)}, features: {len(features)}"

    values = np.array(values)[demo_dim:].tolist()

    important_features = []
    for i, feature in enumerate(features):
        important_features.append((feature, values[i], attns[i]))
    important_features.sort(key=lambda x: x[2], reverse=True)
    return important_features[:3]


def get_var_desc(var: float):
    if var > 0:
        return round(var * 100, 2)
    else:
        return round(-var * 100, 2)


def get_trend_desc(var: float):
    if var > 0:
        return "increased"
    else:
        return "decreased"


def get_recommended_trend_desc(var: float):
    if var > 0:
        return "decrease"
    else:
        return "increase"


def get_range_desc(key: str, var: float):
    if key in ["Weight", "Appetite"]:
        return ""
    if var < medical_standard[key][0]:
        return f"the value is lower than normal range by {round((medical_standard[key][0] - var) / medical_standard[key][0] * 100, 2)}%"
    elif var > medical_standard[key][1]:
        return f"the value is higher than normal range by {round((var - medical_standard[key][1]) / medical_standard[key][1] * 100, 2)}%"
    else:
        return "the value is within the normal range"


def get_mean_desc(var: str, mean: float):
    if var < mean:
        return f"{round((mean - var) / mean * 100, 0)}% lower"
    elif var > mean:
        return f"{round((var - mean) / mean * 100, 0)}% higher"


def get_death_desc(risk: float):
    if risk < 0.5:
        return "a low level"
    elif risk < 0.7:
        return "a high level"
    else:
        return "an extremely high level"


def get_distribution(data, values):
    arr = np.sort(np.array(values))
    index = np.searchsorted(arr, data, side='right')
    rank = index / len(arr) * 100
    if rank < 40:
        return "at the bottom 40% levels"
    elif rank < 70:
        return "at the middle 30% levels"
    else:
        return "at the top 30% levels"


def main():
    parser = argparse.ArgumentParser(description="Generate query for LLM")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "tjh", "esrd", "obstetrics"], help="Specify dataset name: mimic-iv or tjh or esrd or obstetrics")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb", "los"], help="Prediction task: mortality or readmission or sptb or los")
    parser.add_argument("--models", "-m", nargs='+', default=["AdaCare", "ConCare", "RETAIN"],
                       help="DL models to use for generating query")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"],
                       help="Modality of the dataset: ehr or note or mm")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    models = args.models
    modality = args.modality
    print(f"Dataset: {dataset}, Task: {task}, Models: {models}, Modality: {modality}")

    split = "split"
    if dataset == 'tjh':
        demo_dim = 2
        lab_dim = 73
    elif args.dataset == 'mimic-iv':
        demo_dim = 2
        lab_dim = 42
    elif args.dataset == 'esrd':
        demo_dim = 0
        lab_dim = 17
    elif args.dataset == 'obstetrics':
        demo_dim = 0
        lab_dim = 32
        split = "solo"
    else:
        raise ValueError("Unsupported dataset. Choose either 'tjh' or 'mimic-iv' or 'esrd' or 'obstetrics'.")

    dataset_root = "my_datasets/ehr"
    results_root = "logs"

    ids, _, x_llm_ts, x_note, labels, missing_masks, record_times, labtest_features, _, basic_info, survival_stats, dead_stats = load_dataset(dataset_root, dataset, task, split)
    preds = {}
    attns = {}
    for model in models:
        pred, attn = load_training_results(results_root, dataset, task, model)
        preds[model] = pred
        attns[model] = attn
    query_list = []

    for i, id in enumerate(ids):
        ehr_context = "Here is multivariate time-series electronic health record data of the patient, a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature.\n"
        ehr_context += format_input_ehr(x_llm_ts[i], dataset, labtest_features, missing_masks[i], record_times[i], demo_dim)

        basic_data = basic_info[id]
        preds_item = []
        important_features_item = []
        for model in models:
            preds_item.append(preds[model][i])
            important_features_item.append(process_important_features(x_llm_ts[i][-1], attns[model][i], labtest_features, demo_dim))
        basic_context, last_visit_contexts = generate_prompt(dataset, task, models, basic_data, preds_item, important_features_item, survival_stats, dead_stats)

        ehr_contexts = [basic_context + ehr_context + last_visit_context for last_visit_context in last_visit_contexts]
        note_context = f"Here is the patient's clinical note data.\n{x_note[i]}\n" if x_note is not None else ""

        if modality == "ehr":
            query = ehr_contexts
        elif modality == "note":
            query = note_context
        elif modality == "mm":
            query = [note_context + ehr_context for ehr_context in ehr_contexts]
        query_list.append({
            'qid': id,
            'question': query,
            'ground_truth': labels[i][-1]
        })

    with open(os.path.join(dataset_root, f"{dataset}/processed/{modality}_{task}_test.json"), "w") as f:
        json.dump(query_list, f, indent=4)


if __name__ == "__main__":
    main()