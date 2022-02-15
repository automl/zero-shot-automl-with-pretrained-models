import os
import sys
import pickle
import math
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())
from src.available_datasets import all_datasets, GROUPS

import torch

if torch.cuda.is_available():
    import tensorflow as tf
    tf.enable_eager_execution()

def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

verbosity_level = "INFO"
logger = get_logger(verbosity_level)

def load_autodl_dataset(dataset_dir, verbose = False):

    from src.competition.ingestion_program import data_io
    from src.competition.ingestion_program.dataset import AutoDLDataset  # THE class of AutoDL datasets
    from src.competition.scoring_program.score import get_solution

    ###########################################
    #### COPIED FROM THE INGESTION PROGRAM ####
    ###########################################

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith(".data")]

    if len(datanames) != 1:
        raise ValueError(
            "{} datasets found in dataset_dir={}!\n".format(len(datanames), dataset_dir) +
            "Please put only ONE dataset under dataset_dir."
        )

    basename = datanames[0]
    
    ##### Begin creating training set and test set #####
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))

    if verbose:
        logger.info("************************************************")
        logger.info("******** Processing dataset " + basename[:-5].capitalize() + " ********")
        logger.info("************************************************")

        logger.info('Length of the training set: %d' % D_train.get_metadata().size())
        logger.info('Length of the test set: %d' % D_test.get_metadata().size())

    return D_train, D_test, get_solution(dataset_dir)

def get_dataset_hwc(train_dataset, test_dataset):
    metadata_train = train_dataset.get_metadata()
    metadata_test = test_dataset.get_metadata()
    h_train, w_train, c_train = metadata_train.get_tensor_size()
    h_test, w_test, c_test = metadata_test.get_tensor_size()

    h = -1 if h_train == -1 or h_test == -1 else max(h_train, h_test)
    w = -1 if w_train == -1 or w_test == -1 else max(w_train, w_test)
    c = max(c_train, c_test)

    return h, w, c

def pad_images(images, max_h, max_w):
    padded_images = []
    for img in images:
        img = torch.from_numpy(img)
        _, img_h, img_w = img.size()  
        w_pad_1 = math.ceil((max_w-img_w)/2)
        w_pad_2 = math.floor((max_w-img_w)/2)
        h_pad_1 = math.ceil((max_h-img_h)/2)
        h_pad_2 = math.floor((max_h-img_h)/2)
        padder = torch.nn.ZeroPad2d((w_pad_1, w_pad_2, h_pad_1, h_pad_2))
        padded_images.append(padder(img).numpy())

    return padded_images

def interpolate_images(images, max_h, max_w):
    int_images = []
    for img in images:
        img = torch.from_numpy(img).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size = (max_h, max_w), mode = 'bilinear')
        int_images.append(img.squeeze(0).numpy())

    return int_images

def autodl2torch(dataset, size_limit = 224):

    h, w, c = dataset.get_metadata().get_tensor_size()
    print(h, w, c)
    
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset.get_dataset().take(-1)):

        image = image.numpy()
        label = label.numpy()

        if c == 1:
            image = np.stack((image.squeeze(-1),)*3, axis=-1)
        images.append(image.squeeze(0).transpose(2, 0, 1))
        labels.append(np.argmax(label))

    if len(images) > 1e5:
        pairs = random.sample(list(zip(images, labels)), 100000)
        images, labels = zip(*pairs)

    num_classes=max(labels)+1

    images = interpolate_images(images, size_limit, size_limit)
    
    images = np.array(images)
    labels = np.array(labels)
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(images, labels)

    return dataset, num_classes

def autodl2torch_wrapper(datasets_main_dir, dataset_names, converted_datasets_dir, n_augmentations = 15):
    os.makedirs(converted_datasets_dir, exist_ok = True)

    for n in range(n_augmentations):
        dataset_dirs = [os.path.join(datasets_main_dir, str(n), dataset_name) for dataset_name in dataset_names]
        for dataset_name, dataset_dir in zip(dataset_names, dataset_dirs):

            dataset_id = str(n)+'-'+dataset_name
            logger.info(f"Converting {dataset_id}")
            
            train_dataset, _, _ = load_autodl_dataset(dataset_dir)
            dataset, num_classes = autodl2torch(train_dataset, size_limit = 48)

            tensor_dict = {'dataset': dataset, 'num_classes': num_classes}
            torch.save(tensor_dict, os.path.join(converted_datasets_dir, dataset_id+'.pt'))


def load_torch_dataset(dataset_dir):
    tensor_dict = torch.load(dataset_dir)
    dataset = tensor_dict['dataset']
    num_classes = tensor_dict['num_classes']

    return dataset, num_classes


def autodl2numpy(dataset, test_solution = None, **opts):

    size_limit = opts['size_limit'] if 'size_limit' in opts else 224
    max_samples = opts['max_samples'] if 'max_samples' in opts else 1e5
    h = opts['shape'][0] if 'shape' in opts else -1
    w = opts['shape'][1] if 'shape' in opts else -1
    c = opts['shape'][2] if 'shape' in opts else 3
    
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset.get_dataset().take(-1)):

        image = image.numpy()
        if test_solution is not None:
            label = test_solution[i]

        if c == 1:
            image = np.stack((image.squeeze(-1),)*3, axis=-1)

        images.append(image.squeeze(0).transpose(2, 0, 1))
        labels.append(np.argmax(label))
    
    if len(images) > max_samples:
        pairs = random.sample(list(zip(images, labels)), int(max_samples))
        images, labels = zip(*pairs)

    # resize them if too large to avoid memory issues
    images = interpolate_images(images, size_limit, size_limit)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_numpy_dataset(dataset_dir):
    train_x = np.load(os.path.join(dataset_dir,'train_x.npy'))
    train_y = np.load(os.path.join(dataset_dir,'train_y.npy'))
    test_x = np.load(os.path.join(dataset_dir,'test_x.npy'))
    test_y = np.load(os.path.join(dataset_dir,'test_y.npy'))

    return (train_x, train_y), (test_x, test_y)


def convert_metadata_to_df(metadata):
    k, v = list(metadata.items())[0]
    columns = sorted(v.keys())
    columns_edited = False

    features_lists = []
    indices = []

    for key, values_dict in sorted(metadata.items()):
        indices.append(key)
        feature_list = [values_dict[k] for k in sorted(values_dict.keys())]

        # below loop flattens feature list since there are tuples in it &
        # it extends columns list accordingly
        for i, element in enumerate(feature_list):
            if type(element) is tuple:
                # convert tuple to single list elements
                slce = slice(i, i + len(element) - 1)

                feature_list[slce] = list(element)

                if not columns_edited:
                    columns_that_are_tuples = columns[i]
                    new_columns = [
                        columns_that_are_tuples + "_" + str(i) for i in range(len(element))
                    ]
                    columns[slce] = new_columns
                    columns_edited = True

        features_lists.append(feature_list)

    return pd.DataFrame(features_lists, columns=columns, index=indices)


def dump_meta_features_df_and_csv(meta_features, n_augmentations, output_path, file_name="meta_features"):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    if not isinstance(meta_features, pd.DataFrame):
        df = convert_metadata_to_df(meta_features)
    else:
        df = meta_features

    df.to_csv((output_path / file_name).with_suffix(".csv"))
    df.to_pickle((output_path / file_name).with_suffix(".pkl"))

    print("meta features data dumped to: {}".format(output_path))

def dataset_info(dataset):
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset.get_dataset().take(-1)):
        image = image.numpy().squeeze(0)
        images.append(image)
        labels.append(np.argmax(label))

    if len(images) > 1e4:
        pairs = random.sample(list(zip(images, labels)), int(1e4))
        images, labels = zip(*pairs)

    num_classes=max(labels)+1

    images = np.array(images)
    labels = np.array(labels)

    image_shapes = [img.shape[:2] for img in images]
    image_shapes = np.stack(image_shapes)

    return image_shapes

def get_real_image_sizes(datasets_main_dir, dataset_names, n_augmentations = 15):
    
    per_dataset_size_dict = dict()
    for n in range(n_augmentations):
        
        dataset_dirs = [os.path.join(datasets_main_dir, str(n), dataset_name) for dataset_name in dataset_names]
        for dataset_name, dataset_dir in zip(dataset_names, dataset_dirs):
        
            train_dataset, test_dataset, test_solution = load_autodl_dataset(dataset_dir)
            h, w, c = train_dataset.get_metadata().get_tensor_size()
            if h == -1 or w == -1:
                image_shapes = dataset_info(train_dataset)
                _min = np.amin(image_shapes, 0)
                _max = np.amax(image_shapes, 0)
                _mean = image_shapes.mean(0)

                dataset_id = str(n)+'-'+dataset_name
                per_dataset_size_dict[dataset_id] = {"min": _min, "max": _max, "mean": _mean}
                print(f"Dataset {dataset_id}")
                print(f"Minimum and maximum image shapes: {_min}, {_max} | Mean HW: {_mean}")

    return per_dataset_size_dict

if __name__ == '__main__':
    
    datasets_main_dir = '../../data/datasets/'
    dataset_names = GROUPS['all']
    N_AUGMENTATIONS = 15
    
    # Convert training set from AutoDL to Torch tensor format
    converted_datasets_dir = "../../data/augmented_train_datasets_torch"
    autodl2torch_wrapper(datasets_main_dir, dataset_names, converted_datasets_dir)
    

    # Get real dataset image sizes for variable size datasets
    get_real_image_sizes(datasets_main_dir, dataset_names)

    