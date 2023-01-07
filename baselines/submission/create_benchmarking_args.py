import os
from time import time
from random import shuffle

solutions = ['Random-selection', "Single-best", "Oracle",
             'DeepWisdom', 'DeepBlueAI', 'PASA_NJU', 'ZAP-AS', 'ZAP-HPO', 
             "ZAP-HPO-D25", "ZAP-HPO-D50", "ZAP-HPO-D75"]

datasets_main_dir = "../data/datasets" 
n_augmentations = 15

all_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette', 'emnist_byclass', 
                'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 
                'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']

all_paths = [os.path.join(datasets_main_dir, str(n), dataset_name) for n in range(n_augmentations) for dataset_name in all_datasets]
all_names = [str(n)+"-"+dataset_name for n in range(n_augmentations) for dataset_name in all_datasets]

all_commands = []
for s in solutions:
    for p, n in zip(all_paths, all_names):
        for r in range(10):
            result_dir_path = os.path.join("../data/ZAP_benchmark_scoring_output", s, n)+"-"+str(r)
            cmd = ' '.join(["--code_dir", s+"_submission","--dataset_dir", p, "--result_dir", result_dir_path])
            all_commands.append(cmd)
    
shuffle(all_commands)
bs = 10500
num_arg_batches = int(len(all_commands)//bs)
for b in range(num_arg_batches):
    f = open(f"submission/benchmarking_batch_{b}.args", "w+")
    for cmd in all_commands[b*bs:(b+1)*bs]:
        f.write(cmd+"\n")
    f.close()


