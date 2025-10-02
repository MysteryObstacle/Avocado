import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import json
import gzip
import click


def save_parquet(df, path):
    df.to_parquet(path, index=False)


def save_train(df, path_dir):
    path = path_dir / "train.parquet"
    save_parquet(df, path)


def save_test(df, path_dir):
    path = path_dir / "test.parquet"
    save_parquet(df, path)


def save_valid(df, path_dir):
    path = path_dir / "valid.parquet"
    save_parquet(df, path)


def create_train_valid_test_for_task(df, label_col, test_size, valid_size, under_sampling, data_dir_path):
    task_df = df.dropna(subset=[label_col])[['feature', label_col]].rename(columns={label_col: 'label'})

    train_df, valid_df, test_df = split_train_valid_test(task_df, test_size, valid_size, under_sampling)

    save_train(train_df, data_dir_path)
    save_valid(valid_df, data_dir_path)
    save_test(test_df, data_dir_path)


def split_train_valid_test(df, test_size, valid_size, under_sampling_train=True):
    # remove labels with less than 100 samples
    value_counts = df['label'].value_counts()
    valid_labels = value_counts[value_counts >= 20].index
    df = df[df['label'].isin(valid_labels)]

    print(df['label'].value_counts())

    features = df['feature']
    labels = df['label']

    features_train_valid, features_test, labels_train_valid, labels_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels if under_sampling_train else None
    )

    valid_size_adjusted = valid_size / (1 - test_size)
    features_train, features_valid, labels_train, labels_valid = train_test_split(
        features_train_valid, labels_train_valid, test_size=valid_size_adjusted,
        stratify=labels_train_valid if under_sampling_train else None
    )

    if under_sampling_train:
        df_train = pd.DataFrame({'feature': features_train, 'label': labels_train})
        df_valid = pd.DataFrame({'feature': features_valid, 'label': labels_valid})
        df_test = pd.DataFrame({'feature': features_test, 'label': labels_test})

        min_label_count = df_train['label'].value_counts().min()
        df_train = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(min(min_label_count, len(x))))

        return df_train, df_valid, df_test
    else:
        return (
            pd.DataFrame({'feature': features_train, 'label': labels_train}),
            pd.DataFrame({'feature': features_valid, 'label': labels_valid}),
            pd.DataFrame({'feature': features_test, 'label': labels_test})
        )


def print_df_label_distribution(path):
    df = pd.read_parquet(path)
    print(df['label'].value_counts())


@click.command()
@click.option(
    "-r",
    "--root",
    help="root directory of the original files",
    required=True,
)
@click.option(
    "-t",
    "--test_size",
    help="test size",
    default=0.15,
)
@click.option(
    "-v",
    "--valid_size",
    help="validation size",
    default=0.15,
)
@click.option(
    "-u",
    "--under_sampling",
    help="under sampling",
    default=False,
)
def main(root, test_size, valid_size, under_sampling):
    source_data_dir_path = Path(os.path.join(root, 'deal'))
    target_data_dir_path = Path(os.path.join(root, 'data'))

    target_data_dir_path.mkdir(parents=True, exist_ok=True)

    # read data
    file_paths = []
    for root, dirs, files in os.walk(source_data_dir_path):
        file_paths.extend(list(Path(root).glob('*.json.gz')))

    data = []
    for file_path in file_paths:
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Print the total number of rows
    print(f"Total number of rows: {len(df)}")

    # prepare data for application classification and traffic classification
    create_train_valid_test_for_task(
        df=df,
        label_col="attack_label",
        test_size=test_size,
        valid_size=valid_size,
        under_sampling=under_sampling,
        data_dir_path=target_data_dir_path
    )

    # stats
    print("Training data distribution:")
    print_df_label_distribution(target_data_dir_path / "train.parquet")
    print("Testing data distribution:")
    print_df_label_distribution(target_data_dir_path / "test.parquet")
    print("Validation data distribution:")
    print_df_label_distribution(target_data_dir_path / "valid.parquet")


if __name__ == "__main__":
    main()
