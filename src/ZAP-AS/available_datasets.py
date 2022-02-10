from pathlib import Path
import yaml
import os

N_AUGMENTATIONS = 15

all_datasets = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette', 'emnist_byclass', 
                'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 
                'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']

# Same as the AutoFolio inner CV splits
train_splits = [['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 'cmaterdb_bangla', 'mnist'], 
                ['horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 'cycle_gan_apple2orange', 'cycle_gan_maps', 'imagenette'], 
                ['emnist_byclass', 'svhn_cropped', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 'cycle_gan_ukiyoe2photo'], 
                ['cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 'emnist_balanced'], 
                ['cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs', 'imagenet_resized_32x32', 'omniglot', 'colorectal_histology']]

# Domains
objects = ['cifar100', 'cifar10', 'horses_or_humans', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
           'cycle_gan_apple2orange', 'imagenette', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 
           'tf_flowers', 'cassava', 'fashion_mnist', 'cars196', 'cats_vs_dogs', 'imagenet_resized_32x32',
           ]

ocr = ['cmaterdb_devanagari', 'cmaterdb_bangla', 'mnist', 'kmnist', 'emnist_byclass', 'emnist_mnist', 
       'cmaterdb_telugu', 'emnist_balanced', 'omniglot', 'svhn_cropped']

medical = ['colorectal_histology', 'malaria']
aerial = ['uc_merced', 'cycle_gan_maps', 'eurosat_rgb']
other = ['cycle_gan_ukiyoe2photo', 'cycle_gan_vangogh2photo', 'cycle_gan_summer2winter_yosemite', 'cycle_gan_iphone2dslr_flower']


GROUPS = {'all': all_datasets, 
          'objects': objects,
          'ocr': ocr,
          'medical': medical,
          'aerial' : aerial,
          'other': other
        }

if __name__ == '__main__':

    for group, elements in GROUPS.items():
        print('Dataset group {} contains {} dataset(s). These are ->'.format(group, len(elements)))
        print('\n'.join(elements))
        print('='*60)
