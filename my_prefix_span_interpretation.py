import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random
import os
from glob import glob
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import KBinsDiscretizer
import prefixspan
from natsort import natsorted
import matplotlib.pyplot as plt
import gc



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

column_path = '../data/battery_brand3/column.pkl'
column_names = torch.load(column_path, weights_only=False)
tables_path = '../data/battery_brand3'


def get_car_label(car_id, metadata):
    for (c_id, seg_id), meta in metadata.items():
        if c_id == car_id:
            if meta['label'] == '00':
                return 0
            else:
                return 1
    return None


def load_all_snippets(folder_path):
    car_data = defaultdict(lambda: defaultdict(list))
    all_metadata = {}

    for pkl_file in Path(folder_path).glob('*.pkl'):
        data_tuple = torch.load(pkl_file, weights_only=False)
        data_list, metadata = data_tuple

        columns = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt',
                   'max_temp', 'min_temp', 'timestamp']
        df = pd.DataFrame(data_list, columns=columns)

        car_id = metadata['car']
        segment_id = metadata['charge_segment']

        car_data[car_id][segment_id].append(df)
        all_metadata[(car_id, segment_id)] = metadata

    return car_data, all_metadata


def sort_segments_by_id(car_data):
    sorted_car_data = defaultdict(lambda: defaultdict(list))

    for car_id in car_data:
        segment_ids = natsorted(car_data[car_id].keys())

        for segment_id in segment_ids:
            snippets = car_data[car_id][segment_id]

            sorted_snippets = sorted(
                snippets,
                key=lambda df: df['timestamp'].iloc[0]
            )

            sorted_car_data[car_id][segment_id] = sorted_snippets

    return sorted_car_data


def extract_features_from_snippet(df):
    features = {}

    # Напряжение
    features['volt_mean'] = df['volt'].mean()
    features['volt_std'] = df['volt'].std()
    features['volt_slope'] = (df['volt'].iloc[-1] - df['volt'].iloc[0]) / len(df)
    features['volt_range'] = df['volt'].max() - df['volt'].min()

    # Ток
    features['current_mean'] = df['current'].mean()
    features['current_abs_mean'] = abs(df['current']).mean()
    features['current_std'] = df['current'].std()
    features['current_slope'] = (df['current'].iloc[-1] - df['current'].iloc[0]) / len(df)

    # SOC
    features['soc_start'] = df['soc'].iloc[0]
    features['soc_end'] = df['soc'].iloc[-1]
    features['soc_change'] = df['soc'].iloc[-1] - df['soc'].iloc[0]
    features['soc_std'] = df['soc'].std()

    # Температура
    features['max_temp_mean'] = df['max_temp'].mean()
    features['min_temp_mean'] = df['min_temp'].mean()
    features['temp_spread'] = df['max_temp'].max() - df['min_temp'].min()
    features['temp_imbalance'] = (df['max_temp'] - df['min_temp']).mean()
    features['temp_slope'] = (df['max_temp'].iloc[-1] - df['max_temp'].iloc[0]) / len(df)

    # Напряжение отдельных ячеек
    features['cell_imbalance_mean'] = (df['max_single_volt'] - df['min_single_volt']).mean()
    features['cell_imbalance_max'] = (df['max_single_volt'] - df['min_single_volt']).max()
    features['cell_imbalance_std'] = (df['max_single_volt'] - df['min_single_volt']).std()

    # Длительность
    features['duration'] = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

    return features


def train_discretizers_on_normal_snippets(car_data, metadata):
    all_features = []
    normal_cars_count = 0
    normal_snippets_count = 0

    for car_id in car_data:
        label = get_car_label(car_id, metadata)

        if label == 0:
            normal_cars_count += 1
            for segment_id in car_data[car_id]:
                for snippet_df in car_data[car_id][segment_id]:
                    features = extract_features_from_snippet(snippet_df)
                    all_features.append(features)
                    normal_snippets_count += 1

    if len(all_features) == 0:
        print("Нет нормальных сниппетов для обучения")
        return None

    print(f"Найдено {normal_cars_count} нормальных автомобилей, {normal_snippets_count} сниппетов")

    # Преобразуем в DataFrame
    features_df = pd.DataFrame(all_features)

    # Обучаем дискретизаторы для каждого признака (только для ключевых)
    key_features = ['volt_slope', 'current_abs_mean', 'current_slope',
                    'soc_change', 'temp_spread', 'temp_imbalance',
                    'cell_imbalance_max', 'cell_imbalance_mean']

    discretizers = {}
    for col in key_features:
        if col in features_df.columns:
            # Quantile binning - 3 категории: LOW (0), NORMAL (1), HIGH (2)
            disc = KBinsDiscretizer(
                n_bins=3,
                encode='ordinal',
                strategy='quantile'
            )
            disc.fit(features_df[[col]])
            discretizers[col] = disc

    print(f"Обучено {len(discretizers)} дискретизаторов")
    return discretizers


def discretize_value(value, discretizer):
    feature_name = discretizer.feature_names_in_[0]
    df = pd.DataFrame([[value]], columns=[feature_name])
    return int(discretizer.transform(df)[0][0])


def snippet_to_event(df, discretizers=None, state_map=None):
    if state_map is None:
        state_map = {0: 'LOW', 1: 'NORMAL', 2: 'HIGH'}

    features = extract_features_from_snippet(df)

    event_parts = []

    if discretizers:
        for feat_name, disc in discretizers.items():
            if feat_name in features:
                try:
                    state = discretize_value(features[feat_name], disc)
                    if state != 1:
                        short_name = {
                            'volt_slope': 'V',
                            'current_abs_mean': 'I',
                            'current_slope': 'I_SLOPE',
                            'soc_change': 'SOC',
                            'temp_spread': 'T_SPREAD',
                            'temp_imbalance': 'T_IMB',
                            'cell_imbalance_max': 'CELL_MAX',
                            'cell_imbalance_mean': 'CELL_MEAN'
                        }.get(feat_name, feat_name.upper())

                        event_parts.append(f"{short_name}_{state_map[state]}")
                except Exception as e:
                    continue
    else:
        print("Дискретизаторы не обучены")
        exit()

    if len(event_parts) == 0:
        return 'NORMAL'

    return ','.join(sorted(event_parts))


def build_car_sequence(car_id, car_data, metadata, discretizers=None):
    sequence = []
    timestamps = []
    segment_ids = []
    snippet_indices = []

    sorted_segments = natsorted(car_data[car_id].keys())

    for seg_id in sorted_segments:
        snippets = car_data[car_id][seg_id]

        sorted_snippets = sorted(
            snippets,
            key=lambda df: df['timestamp'].iloc[0] if len(df) > 0 else 0
        )

        for snip_idx, snippet_df in enumerate(sorted_snippets):
            event = snippet_to_event(snippet_df, discretizers)

            sequence.append(event)
            timestamps.append(snippet_df['timestamp'].iloc[0] if len(snippet_df) > 0 else None)
            segment_ids.append(seg_id)
            snippet_indices.append(snip_idx)


    return {
        'car_id': car_id,
        'sequence': sequence,
        'timestamps': timestamps,
        'segment_ids': segment_ids,
        'snippet_indices': snippet_indices,
        'label': get_car_label(car_id, metadata)
    }


def build_all_sequences(car_data, metadata, discretizers=None):
    car_sequences = {}
    normal_sequences = []
    anomaly_sequences = []

    for car_id in car_data:
        seq_data = build_car_sequence(car_id, car_data, metadata, discretizers)
        car_sequences[car_id] = seq_data

        if seq_data['label'] == 0:
            normal_sequences.append(seq_data['sequence'])
        else:
            anomaly_sequences.append(seq_data['sequence'])

    print(f"Построено {len(car_sequences)} последовательностей")
    print(f"Нормальных: {len(normal_sequences)}")
    print(f"Аномальных: {len(anomaly_sequences)}")

    if anomaly_sequences:
        lengths = [len(seq) for seq in anomaly_sequences]
        print(f"Аномальные: мин={min(lengths)}, макс={max(lengths)}, сред={np.mean(lengths):.2f}")

    return car_sequences, normal_sequences, anomaly_sequences


def find_unique_anomaly_patterns(anomaly_sequences, normal_sequences, min_support=2):
    def clean_sequences(sequences, max_len_keep=30):
        cleaned = []
        for seq in sequences:
            filtered = [e for e in seq if e != 'NORMAL']

            if len(filtered) > max_len_keep:
                filtered = filtered[-max_len_keep:]

            unique = []
            for e in filtered:
                if not unique or unique[-1] != e:
                    unique.append(e)

            if len(unique) >= 2:
                cleaned.append(unique)
        return cleaned

    anomaly_filtered = clean_sequences(anomaly_sequences, max_len_keep=30)
    normal_filtered = clean_sequences(normal_sequences, max_len_keep=30)

    print(f"Аномальных последовательностей (>=2 событий): {len(anomaly_filtered)}")
    print(f"Нормальных последовательностей (>=2 событий): {len(normal_filtered)}")

    if len(anomaly_filtered) == 0:
        print("Нет аномальных последовательностей для анализа")
        return []

    ps_anomaly = prefixspan.PrefixSpan(anomaly_filtered)
    anomaly_results = ps_anomaly.frequent(minsup=2, closed=True)
    anomaly_patterns = {}
    for support, pattern in anomaly_results:
        anomaly_patterns[tuple(pattern)] = support

    def filter_by_length(patterns, min_len=2, max_len=5):
        return {p: s for p, s in patterns.items()
                if min_len <= len(p) <= max_len}

    anomaly_patterns = filter_by_length(anomaly_patterns, min_len=2, max_len=4)

    def filter_consecutive_duplicates(patterns):
        result = {}
        for pattern, support in patterns.items():
            has_duplicate = False
            for i in range(len(pattern) - 1):
                if pattern[i] == pattern[i + 1]:
                    has_duplicate = True
                    break
            if not has_duplicate:
                result[pattern] = support
        return result

    anomaly_patterns = filter_consecutive_duplicates(anomaly_patterns)
    print(f"Найдено {len(anomaly_patterns)} паттернов в аномальных данных")


    if normal_filtered:
        ps_normal = prefixspan.PrefixSpan(normal_filtered)
        normal_results = ps_normal.frequent(minsup=min_support, closed=True)  # более строго
        normal_patterns = {}
        for support, pattern in normal_results:
            normal_patterns[tuple(pattern)] = support

        normal_patterns = filter_by_length(normal_patterns, min_len=2, max_len=4)
        normal_patterns = filter_consecutive_duplicates(normal_patterns)
        print(f"Найдено {len(normal_patterns)} паттернов в нормальных данных")
    else:
        normal_patterns = {}

    unique_patterns = {}
    for pattern, support in anomaly_patterns.items():
        if pattern not in normal_patterns:
            if any(event != 'NORMAL' for event in pattern):
                unique_patterns[pattern] = support

    def remove_subpatterns(patterns):
        def is_subsequence(sub, seq):
            it = iter(seq)
            return all(item in it for item in sub)

        pattern_list = list(patterns.items())
        result = {}

        for i, (pattern, support) in enumerate(pattern_list):
            is_sub = False
            for j, (longer, _) in enumerate(pattern_list):
                if i != j and len(longer) > len(pattern):
                    if is_subsequence(pattern, longer):
                        is_sub = True
                        break
            if not is_sub:
                result[pattern] = support

        return result

    unique_patterns = remove_subpatterns(unique_patterns)

    sorted_patterns = sorted(unique_patterns.items(), key=lambda x: x[1], reverse=True)
    sorted_patterns = sorted_patterns[:100]

    print(f"Уникальных паттернов: {len(sorted_patterns)}")

    gc.collect()
    return sorted_patterns


def save_patterns_report(unique_patterns, car_sequences):
    output_file = f'../data/{brand}/patterns_report.xlsx'
    report = []

    for pattern, support in unique_patterns[:100]:
        cars_with_pattern = []
        segments_info = []

        for car_id, seq_data in car_sequences.items():
            if seq_data['label'] == 1:
                seq = seq_data['sequence']
                if all(e in seq for e in pattern):
                    cars_with_pattern.append(car_id)
                    segs = []
                    for idx, e in enumerate(seq_data['sequence']):
                        if e in pattern and idx < len(seq_data['segment_ids']):
                            seg_id = seq_data['segment_ids'][idx]
                            if seg_id not in segs and seg_id is not None:
                                segs.append(seg_id)
                    segments_info.append({'car': car_id, 'segments': segs})

        report.append({
            'pattern': ' -> '.join(pattern),
            'pattern_length': len(pattern),
            'support': support,
            'cars_with_pattern': len(cars_with_pattern),
            'example_car': cars_with_pattern[0] if cars_with_pattern else '',
            'example_segments': str(segments_info[0]['segments']) if segments_info else ''
        })

    report_df = pd.DataFrame(report)
    report_df.to_excel(output_file, index=False)
    print(f"\nОтчет сохранен в {output_file}")
    return report_df


def run_full_pipeline(folder_path, min_support=2):
    car_data, metadata = load_all_snippets(folder_path)

    if not car_data:
        print("Ошибка: данные не загружены!")
        return

    car_data = sort_segments_by_id(car_data)
    discretizers = train_discretizers_on_normal_snippets(car_data, metadata)
    car_sequences, normal_sequences, anomaly_sequences = build_all_sequences(
        car_data, metadata, discretizers
    )

    if len(anomaly_sequences) == 0:
        print("Ошибка: нет аномальных автомобилей для анализа!")
        return

    unique_patterns = find_unique_anomaly_patterns(
        anomaly_sequences, normal_sequences, min_support=min_support
    )

    if not unique_patterns:
        print("Паттерны не найдены")
        return

    report_df = save_patterns_report(unique_patterns, car_sequences)


    return {
        'car_data': car_data,
        'metadata': metadata,
        'car_sequences': car_sequences,
        'normal_sequences': normal_sequences,
        'anomaly_sequences': anomaly_sequences,
        'unique_patterns': unique_patterns,
        'report_df': report_df
    }


if __name__ == "__main__":
    brand = 'battery_brand3'
    FOLDER_PATH = f'../data/{brand}/data'

    results = run_full_pipeline(
        folder_path=FOLDER_PATH,
        min_support=3,
    )

    if results:
        print(f"Найдено паттернов: {len(results['unique_patterns'])}")
    else:
        print('Паттерны не найдены')

