import os
import sys
import yaml
from pathlib import Path
import pandas as pd

sys.path.append(os.getcwd())

from loader import bool_hps, categorical_hps, numerical_hps


N_AUGMENTATIONS = 15
all_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari',
                'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades',
                'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette',
                'emnist_byclass',
                'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers',
                'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria',
                'eurosat_rgb',
                'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite',
                'cats_vs_dogs']

HP_NAMES = numerical_hps+bool_hps+categorical_hps

ENCODE_ARCH = {'ResNet18': [1, 0, 0, 0], 
               'efficientnetb0': [0, 1, 0, 0], 
               'efficientnetb1': [0, 0, 1, 0],
               'efficientnetb2': [0, 0, 0, 1]}

ENCODE_OPTIM = {'SGD': [0, 0, 1], 
                'Adam': [0, 1, 0], 
                'AdamW': [1, 0, 0]}

ENCODE_SIMPLE_MODEL = {'SVC': [0, 0, 0, 1], 
                       'NuSVC': [0, 1, 0, 0], 
                       'RF': [0, 0, 1, 0], 
                       'LR': [1, 0, 0, 0]}

ENCODE_SCHED = {'plateau': [0, 1], 
                'cosine': [1, 0]}


def get_config_response(config_dir=Path(__file__).parents[6] / "data/meta_dataset/configs" / "kakaobrain_optimized_per_icgen_augmentation",
                        response_file=Path(__file__).parents[6] / "data/meta_dataset/perf_matrix.csv"):
    config_paths = [os.path.join(config_dir, str(aug_n), dataset_name) + '.yaml' for dataset_name in all_datasets for aug_n in
                    range(N_AUGMENTATIONS)]
    response_ids = [str(aug_n) + '-' + dataset_name for dataset_name in all_datasets for aug_n in range(N_AUGMENTATIONS)]
    response = pd.read_csv(response_file, index_col=0)

    return config_paths, response_ids, response

def get_config_paths(config_dir):
    config_paths = [os.path.join(config_dir, str(aug_n), dataset_name) + '.yaml' for dataset_name in all_datasets for aug_n in
                    range(N_AUGMENTATIONS)]

    return config_paths


def create_searchspace(path):
    with open(path) as stream:
        model_config = yaml.safe_load(stream)

    config = model_config['autocv']
    temp_config = dict()
    temp_config.update(config['checkpoints'])
    temp_config.update(config['conditions'])
    temp_config.update(config['dataset'])
    temp_config.update(config['model'])
    temp_config.update(config['optimizer'])
    config = temp_config

    hp_vector = []

    # NUMERICAL
    hp_vector.append(config['early_epoch'])
    hp_vector.append(config['max_inner_loop_ratio'])
    hp_vector.append(config['min_lr'])
    hp_vector.append(config['skip_valid_score_threshold'])
    hp_vector.append(config['test_after_at_least_seconds'])
    hp_vector.append(config['test_after_at_least_seconds_max'])
    hp_vector.append(config['test_after_at_least_seconds_step'])
    hp_vector.append(config['batch_size'])
    hp_vector.append(config['cv_valid_ratio'])
    hp_vector.append(config['max_size'])
    hp_vector.append(config['max_valid_count'])
    hp_vector.append(config['steps_per_epoch'])
    hp_vector.append(config['train_info_sample'])
    hp_vector.append(config['freeze_portion'])
    hp_vector.append(config['lr'])
    hp_vector.append(config['momentum'] if config['type'] == 'SGD' else 0)
    hp_vector.append(config['warm_up_epoch'])
    hp_vector.append(config['warmup_multiplier'])
    hp_vector.append(config['wd'])

    # BOOLEAN
    hp_vector.append(int(config['first_simple_model']))
    hp_vector.append(int(config['amsgrad']) if (config['type'] == 'Adam' or config['type'] == 'AdamW') else 0)
    hp_vector.append(int(config['nesterov']) if config['type'] == 'SGD' else 0)

    # CATEGORICAL
    hp_vector += ENCODE_SIMPLE_MODEL[config['simple_model']] if config['first_simple_model'] else [0, 0, 0, 0]  # D=4
    hp_vector += ENCODE_ARCH[config['architecture']]  # D=4
    hp_vector += ENCODE_SCHED[config['scheduler']] # D=2
    hp_vector += ENCODE_OPTIM[config['type']] # D=3

    return dict(zip(HP_NAMES, hp_vector))

