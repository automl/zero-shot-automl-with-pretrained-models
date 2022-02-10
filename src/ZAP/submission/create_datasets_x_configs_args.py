import itertools as it
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from random import shuffle

import yaml
from src.available_datasets import all_datasets

def construct_command(config, dataset, base_datasets_dir, repeat, configs_path):
    dataset_dir = Path(base_datasets_dir, dataset)
    return "--model_config_name {} --dataset_dir {} --experiment_name {}/{}_{}".format(
        os.path.join(configs_path.name, config), dataset_dir, dataset,
        config.rsplit(".yaml")[0], repeat
    )


def generate_all_commands(configs_path, args_savepath, n_repeats, num_config_subsets, num_dataset_subsets):
    """
    Parameters:
        configs_path: Path to the configs' dir for evaluation. 
        args_savepath: Path to the output argument file.
        n_repeats: Number of random seed repeats for each dataset-config pair evaluation.
        num_config_subsets: Number of configurations subsets to use in the experiment.
        num_dataset_subsets:  Number of dataset subsets to use in the experiment.
    """
    with Path(configs_path.parent, "default.yaml").open() as in_stream:
        config = yaml.safe_load(in_stream)
        print("using {} as default config file".format(str(configs_path.parent) + "/default.yaml"))
    base_datasets_dir = config["cluster_datasets_dir"]

    all_configs = []
    for n in range(num_config_subsets):
        config_path = Path(configs_path, str(n))
        all_configs += [os.path.join(str(n), inaug_path.name) for inaug_path in config_path.glob("*")]

    all_augmented_datasets = [os.path.join(str(n), dataset) for n in range(num_dataset_subsets) for dataset in all_datasets]

    print('Number of configurations: %d' %len(all_configs))
    print('Number of datasets to evaluate: %d' %len(all_augmented_datasets))
 
    commands = [
        construct_command(config, dataset, base_datasets_dir, repeat, configs_path)
        for config, dataset, repeat in it.product(all_configs, all_augmented_datasets, range(n_repeats))
    ]
    shuffle(commands)
    args_savepath.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--configs_path",
        default="../../data/kakaobrain_optimized_per_icgen_augmentation",
        type=Path,
        help="Specifies where the incumbent configurations are stored"
    )
    parser.add_argument(
        "--args_savepath",
        default="per_icgen_augmentation_x_configs.args",
        help="Specifies the name of the args file to be outputted"
    )
    parser.add_argument(
        "--n_repeats",
        default=3,
        type=int,
        help="Specifies how many times one incumbent configuration should be evaluated per dataset"
    )

    parser.add_argument("-ca",
        "--num_config_subsets",
        default=15,
        type=int,
        help="Specifies # of config augmentations"
    )
    parser.add_argument("-da",
        "--num_dataset_subsets",
        default=15,
        type=int,
        help="Specifies # of dataset augmentations"
    )

    args = parser.parse_args()

    args_savepath = Path("submission") / args.args_savepath
    
    generate_all_commands(args.configs_path, args_savepath, args.n_repeats, args.num_config_subsets, args.num_dataset_subsets)

