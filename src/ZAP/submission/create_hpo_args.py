from pathlib import Path
import os
import sys
sys.path.append(os.getcwd())

from src.available_datasets import all_datasets

def generate_all_commands(args_savepath, n_augmentations):
    '''
    '''
    commands = []

    for n in range(n_augmentations):
        for dataset in all_datasets:
            cmd =  "--experiment_name {}/{}".format(str(n), dataset)
            commands.append(cmd)

    args_savepath.write_text("\n".join(commands))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--args_savepath", default="per_icgen_augmentation_hpo.args", help="Output file name")
    parser.add_argument("--n_augmentations", default=15, type=int, help="Specifies the number of augmentations")
    args = parser.parse_args()

    args_savepath = Path("submission") / args.args_savepath
    generate_all_commands(args_savepath, args.n_augmentations)

