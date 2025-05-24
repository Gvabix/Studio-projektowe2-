import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from feature_extraction import extract_all_features


def parse_label_from_filename(filename):
    """
    Extracts label from filename:
    '_real_' -> 1
    '_fake_' -> 0
    """
    if '_real_' in filename:
        return 1
    elif '_fake_' in filename:
        return 0
    raise ValueError(f"Cannot parse label from filename: {filename}")


def parse_file_id(filename):
    """
    Extracts the article ID from the filename.
    Assumes filenames are structured like 'prefix_prefix_articleid.ext'
    """
    whole_name = filename.split('.')[0]
    if '_' in whole_name:
        parts = whole_name.split('_')
        return parts[2] if len(parts) > 2 else parts[-1]
    return whole_name

def group_article_files(data_dir):
    """
    Groups .conll, .merge, and .brackets files by article ID.
    """
    file_groups = defaultdict(dict)

    for fname in os.listdir(data_dir):
        if fname.endswith(('.conll', '.merge', '.brackets')):
            base, ext = fname.rsplit('.', 1)
            file_groups[base][ext] = os.path.join(data_dir, fname)

    return file_groups


def load_dataset(data_dir):
    """
    Loads dataset by extracting features from grouped files.
    """
    file_groups = group_article_files(data_dir)
    print(f"Found {len(file_groups)} article groups")

    X, y, skipped = [], [], 0

    for article_id, files in file_groups.items():
        if not all(ext in files for ext in ['conll', 'merge', 'brackets']):
            print(f"Skipping incomplete files for: {article_id}")
            skipped += 1
            continue

        try:
            features = extract_all_features(files['conll'], files['brackets'])
            label = parse_label_from_filename(article_id)

            
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {article_id}: {e}")
            skipped += 1

    return np.array(X), np.array(y)

def load_dataset_with_id(data_dir):
    """
    Loads dataset by extracting features from grouped files.
    """
    file_groups = group_article_files(data_dir)
    print(f"Found {len(file_groups)} article groups")

    X, y, skipped = [], [], 0

    for article_id, files in file_groups.items():
        if not all(ext in files for ext in ['conll', 'merge', 'brackets']):
            print(f"Skipping incomplete files for: {article_id}")
            skipped += 1
            continue

        try:
            id = parse_file_id(article_id)
            features = extract_all_features(files['conll'], files['brackets'])
            label = parse_label_from_filename(article_id)

            
            id_features = [id] + features  # Add ID as the first feature
            X.append(id_features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {article_id}: {e}")
            skipped += 1

    return np.array(X), np.array(y)


def build_feature_dataframe(X, y):
    """
    Converts feature arrays and labels into a clean DataFrame.
    Handles the relation_counts dict structure.
    """


    base_feature_names = [
         'total_tokens', 'avg_token_len', 'NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'PRP',
        'num_edus', 'num_spans', 'num_nucleus', 'num_satellite',
        'ratio_nucleus', 'ratio_satellite', 'relation_counts'
    ]

    df = pd.DataFrame(X, columns=base_feature_names)

    # Expand relation_counts dictionary into separate columns
    relation_df = pd.json_normalize(df['relation_counts']).fillna(0)
    df = pd.concat([df.drop(columns=['relation_counts']), relation_df], axis=1)

    df['label'] = y
    return df


def save_dataset(df, output_path):
    """
    Saves the dataset to a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dataset saved to {output_path}")





if __name__ == "__main__":
    data_dir = '../data/rst_data'
    output_path = os.path.join(data_dir, '../rst_dataset.pkl')

    X, y = load_dataset(data_dir)
    if len(X) == 0:
        print("No data found. Exiting.")
        exit()

    df = build_feature_dataframe(X, y)

    print(df.head())
    print(f"Final dataset shape: {df.shape}")
    save_dataset(df, output_path)
