import os
import sys
sys.path.append(os.getcwd())
sys.path.append("../")
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from available_datasets import all_datasets
from src.competition.ingestion_program.dataset import AutoDLDataset
from src.utils import dump_meta_features_df_and_csv

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def parse_meta_features(meta_data, type="train"):
    sequence_size, x, y, num_channels = meta_data.get_tensor_shape()
    num_classes = meta_data.get_output_size()
    meta_dict = dict(
        num_classes=num_classes,
        sequence_size=sequence_size,
        resolution=(x, y),
        num_channels=num_channels
    )

    if type == "train":
        meta_dict['num_train'] = meta_data.size()
    else:
        meta_dict['num_test'] = meta_data.size()

    return meta_dict


def load_processed_datasets(dataset_path):
    info = {"proc_dataset_dir": dataset_path, "datasets": all_datasets}
    processed_datasets = load_datasets_processed(info, info["datasets"])
    return processed_datasets


def transform_to_long_matrix(df, length_simple_meta_data, n_samples):
    """ transforms a df with shape
    (n_datasets, n_samples*(length_simple_meta_data+length_nn_meta_data) into
    a df with shape
     (n_datasets*n_samples, length_simple_meta_data+length_nn_meta_data)
    """
    len_nn_features = (len(df.columns) - length_simple_meta_data) // n_samples
    new_df = pd.DataFrame(columns=[np.arange(length_simple_meta_data + len_nn_features)])

    for index, row in df.iterrows():
        simple_meta_data = np.asarray(row[:length_simple_meta_data])
        nn_meta_data = np.asarray(row[length_simple_meta_data:])
        nn_meta_splits = np.split(nn_meta_data, n_samples, axis=0)
        for i in range(n_samples):
            new_index = index + "_{}".format(i)
            new_df.loc[new_index] = np.concatenate([simple_meta_data, nn_meta_splits[i]], axis=0)

    return new_df


def get_meta_features_from_dataset(dataset_path, compute_mean_histogram=True, sess=None):
    full_dataset_path = dataset_path / "{}.data/train".format(dataset_path.name)
    full_dataset_path_test = dataset_path / "{}.data/test".format((dataset_path.name))

    if not full_dataset_path.exists():
        full_dataset_path = dataset_path / "{}.data/train".format((dataset_path.name).lower())
    if not full_dataset_path_test.exists():
        full_dataset_path_test = dataset_path / "{}.data/test".format((dataset_path.name).lower())

    train_dataset = AutoDLDataset(str(full_dataset_path))
    test_dataset = AutoDLDataset(str(full_dataset_path_test))
    meta_data_train = train_dataset.get_metadata()
    meta_data_test = test_dataset.get_metadata()
    parsed_meta_data_train = parse_meta_features(meta_data_train, "train")
    parsed_meta_data_test = parse_meta_features(meta_data_test, "test")

    parsed_meta_data = {**parsed_meta_data_train, **parsed_meta_data_test}

    if compute_mean_histogram:
        try:
            sample_count = meta_data_train.size()
            # shuffle the first 2000 (or sample_count) elements and get 100 samples from this buffer
            buffer = 500 if 500 < sample_count else sample_count
            iterator = train_dataset.get_dataset().shuffle(buffer).make_one_shot_iterator()
            next_element = iterator.get_next()

            #meta_sample = sess.run(next_element[0])
            #min, max = meta_sample.min(), meta_sample.max()
            histograms = []
            for _ in range(100):
                sample = sess.run(next_element[0])
                hist = np.histogram(sample, bins=100)[0]
                #hist = sess.run(tf.compat.v1.histogram_fixed_width(next_element[0], [min, max]))
                histograms.append(hist)

            parsed_meta_data["mean_histogram"] = np.mean(histograms, axis=0)

        except Exception as e:
            print("dataset causes issue: {}".format(dataset_path))
            print(traceback.format_exc())

    return parsed_meta_data


def get_nn_meta_features_from_dataset(dataset_path):
    if isinstance(dataset_path, Path):
        dataset_path = str(dataset_path)

    processed_datasets = load_processed_datasets(dataset_path=dataset_path)
    print("getting features ...")
    print("using data: {}".format(dataset_path))

    train_feature_data = [ds[0].dataset.numpy() for ds in processed_datasets]
    train_feature_data = np.concatenate(train_feature_data, axis=0)
    return train_feature_data


def precompute_meta_features(dataset_path, n_augmentations, output_path, dump_dataframe_csv=True, file_name="meta_features"):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    sess = tf.Session()

    dataset_to_meta_features = dict()
    for n in range(n_augmentations):
        dataset_aug_path = Path(dataset_path, str(n))
        for dataset in dataset_aug_path.iterdir():
            dataset_key = str(n)+'-'+dataset.name
            print('Processing: ', dataset)
            dataset_to_meta_features[dataset_key] = get_meta_features_from_dataset(dataset, compute_mean_histogram=False, sess=sess)

    if dump_dataframe_csv:
        dump_meta_features_df_and_csv(meta_features=dataset_to_meta_features, n_augmentations= n_augmentations, output_path=output_path, file_name=file_name)

    output_path_yaml = output_path / file_name
    output_path_yaml = output_path_yaml.with_suffix(".yaml")
    with output_path_yaml.open("w") as out_stream:
        yaml.dump(dataset_to_meta_features, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", 
        type=int, 
        default=2, 
        help="random seed")

    parser.add_argument(
        "--output_savedir", 
        default="../../data/meta_dataset", 
        type=Path, 
        help=" "
    )

    parser.add_argument(
        "--dataset_dir",
        default="../../data/datasets",
        type=Path,
        help=" "
    )

    parser.add_argument(
        "--n_augmentations",
        default=15,
        type=int,
        help=" "
    )
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    precompute_meta_features(
        args.dataset_dir,
        args.n_augmentations,
        args.output_savedir, 
        dump_dataframe_csv=True
    )
	