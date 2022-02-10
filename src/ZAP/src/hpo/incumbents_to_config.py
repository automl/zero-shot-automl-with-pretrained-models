from pathlib import Path

import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')

import hpbandster.core.result as hpres
import yaml
from src.hpo.utils import construct_model_config
import numpy as np

def incumbent_to_config(experiment_path, default_config_path, output_dir):
    # Read the incumbent
    result = hpres.logged_results_to_HBS_result(str(experiment_path))
    id2conf = result.get_id2config_mapping()
    all_runs = result.get_all_runs()
    if len(all_runs) != 100:
        print(print(experiment_path.name))
    inc_id = result.get_incumbent_id()
    incumbent_config = id2conf[inc_id]['config']

    # Read the default config
    with default_config_path.open() as in_stream:
        default_config = yaml.safe_load(in_stream)

    # Compute and write incumbent config in the format of default_config
    incumbent_config = construct_model_config(incumbent_config, default_config)

    out_config_path = output_dir / "{}.yaml".format(experiment_path.name)
    with out_config_path.open("w") as out_stream:
        yaml.dump(incumbent_config, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--default_config_path",
                    default="src/configs/default.yaml",
                    type=Path,
                    help="Specifies where the default yaml file is")
    parser.add_argument("--output_dir",
                    default="../../data/meta_dataset/configs/kakaobrain_optimized_per_icgen_augmentation",
                    type=Path,
                    help="Specifies where the incumbent configs should be stored e.g. data/configs/experiment_name")
    parser.add_argument("--experiment_group_dir",
                    required=True,
                    type=Path,
                    help="Specifies the path to the bohb working directory of an experiment")
    parser.add_argument("--n_augmentations",
                    default=15,
                    type=int,
                    help="Specifies # of augmentations")

    args = parser.parse_args()

   
    for n in np.arange(args.n_augmentations):
        augmentation_path = Path(args.experiment_group_dir, str(n))
        for experiment_path in augmentation_path.iterdir():
            augmentation_output_dir = Path(args.output_dir, str(n))
            os.makedirs(augmentation_output_dir, exist_ok = True)
            try:
                incumbent_to_config(experiment_path, args.default_config_path, augmentation_output_dir)
            except:
                print(experiment_path.name, " has an issue")

