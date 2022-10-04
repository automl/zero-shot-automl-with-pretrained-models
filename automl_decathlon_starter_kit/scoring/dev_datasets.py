import numpy as np
import pickle
import os
import json
import pandas as pd
import scipy.io
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from collections import Counter
import copy

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets

from typing import Callable, Optional, Union

import sys
import time

from fsd50kutils.load_data import load_fsd50k_dataset
from fsd50kutils.audio_dataset import _collate_fn, _collate_fn_eval

"""
The highest-level wrapper function for all the metadata information
Handles the case of a DecathlonDataset as input, or a Subset if a validation split has been performed
Returns a DecathlonMetadata object
"""


def extract_metadata(dataset: Dataset):
    assert isinstance(dataset, Subset) or isinstance(dataset, DecathlonDataset)
    if isinstance(dataset, Subset):
        """
        The Subset points to the original full Dataset, as such its 'sample_count' is incorrect
        """
        metadata = copy.deepcopy(
            dataset.dataset.get_metadata()
        )  # deepcopy to avoid altering the original metadata
        metadata.set_size(
            len(dataset)
        )  # update sample count for our subset; this should be fine as we won't be subsetting fsd50k
    elif isinstance(dataset, DecathlonDataset):
        metadata = dataset.get_metadata()
    return metadata


class DecathlonMetadata(object):
    def __init__(self, root, dataset_name, split):
        self.dataset_name_ = dataset_name

        with open(
            os.path.join(root, "md", dataset_name, f"{split}_metadata.json"), "r"
        ) as f:
            self.metadata_ = json.load(f)

        self.metadata_["input_shape"] = tuple(self.metadata_["input_shape"])
        self.metadata_["output_shape"] = tuple(self.metadata_["output_shape"])

    def get_dataset_name(self):
        return self.dataset_name_

    def get_tensor_shape(self):
        """get_tensor_size updated with sequence size"""
        return self.metadata_["input_shape"]

    def get_output_shape(self):
        return self.metadata_["output_shape"]

    def set_size(self, value):
        self.metadata_["sample_count"] = value

    def size(self):
        return self.metadata_["sample_count"]

    def get_task_type(self):
        return self.metadata_["task_type"]


"""
General Dataset class; wrapper around Datasets for each of the tasks
Following this are all the task-specific Datasets
Loaded x values are in a standard dimension format: (N, Time, Channel, H, W)

If we decide to anonymize the dev tasks, we'll need a quick change of the dataset names/options to be nondescript
"""


class DecathlonDataset(Dataset):
    def __init__(self, task: str, root: str, split: str):

        self.task = task
        self.root = root
        self.split = split

        self.required_batch_size = None
        self.collate_fn = None

        #
        if task == "spherical":
            self.dataset = Spherical(root=root, split=split)
        elif task == "ninapro":
            self.dataset = NinaPro(root=root, split=split)
        elif task == "fsd50k":
            self.dataset = load_fsd50k_dataset(root=root, split=split)
            self.collate_fn = _collate_fn if split == "train" else _collate_fn_eval
            self.required_batch_size = 64
        elif task == "cosmic":
            self.dataset = PairedDatasetImagePath(root=root, split=split)
        elif task == "ecg":
            self.dataset = ECGDataset(root=root, split=split)
        elif task == "deepsea":
            self.dataset = DeepSEA(root=root, split=split)
        elif task == "navierstokes":
            self.dataset = NavierStokes(root=root, split=split)
        elif task == "nottingham":
            self.dataset = Nottingham(root=root, split=split)
        elif task == "ember":
            self.dataset = Ember(root=root, split=split)
        elif task == "crypto":
            self.dataset = Crypto(root=root, split=split)
        else:
            raise ValueError("Did not recognize the task")

        # Creating metadata
        if task == "fsd50k" or split == "test":
            self.metadata = DecathlonMetadata(root, task, split)
        else:
            self.metadata = DecathlonMetadata(root, task, "train")
            # There was potentially a train/validation split that occurred before creating this dataset object; we need to update the size
            new_size = len(self.dataset)
            self.metadata.set_size(new_size)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def get_metadata(self):
        return self.metadata


# function to one-hot-encode a vector of raw labels
def OHE(raw_labels, n):
    ohe_labels = np.eye(n)[raw_labels]
    return ohe_labels


"""
Spherical
"""


def make_4d(inp: torch.Tensor):
    n_dims = len(inp.size())
    return inp[(None,) * (4 - n_dims)]


class Spherical(Dataset):
    def __init__(
        self, root, split, normalize: bool = True, random_crop: Optional[int] = None
    ):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.transform = None
        self.transform = spherical_transform(normalize, random_crop)

        self.x = np.load(
            os.path.join(root, "processed_data", "spherical", f"x_{split}.npy")
        ).transpose(0, 2, 3, 1)
        self.y = np.load(
            os.path.join(root, "processed_data", "spherical", f"y_{split}.npy")
        )
        self.y = OHE(self.y, 100)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def spherical_transform(normalize: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    transform_list.append(transforms.ToTensor())
    if random_crop:
        transform_list.append(transforms.RandomCrop(random_crop))

    if normalize:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    transform_list.append(make_4d)

    return transforms.Compose(transform_list)


"""
NinaPro
"""


class NinaPro(Dataset):
    def __init__(self, root, split):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.x = np.load(
            os.path.join(root, "processed_data", "ninapro", f"x_{split}.npy")
        )
        self.x = self.x.transpose(0, 2, 1)

        self.y = np.load(
            os.path.join(root, "processed_data", "ninapro", f"y_{split}.npy")
        ).astype(int)
        self.y = OHE(self.y, 18)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx, :][None, :, :, None])
        y = torch.tensor(self.y[idx])

        return x, y


"""
FSD50K
Taken care of in the import at the beginning: from .fsd50kutils.load_data import load_fsd50k_dataset
The load_fsd50k_dataset() function will return the necessary Datasets
"""


"""
Cosmic
"""


class PairedDatasetImagePath(Dataset):
    def __init__(self, root, split, skyaug_min=-0.9, skyaug_max=3, seed=None):

        assert split == "train" or split == "test"

        if seed:
            np.random.seed(seed)

        data_path = os.path.join(root, "processed_data", "cosmic", f"{split}.npy")
        sky_path = os.path.join(root, "processed_data", "cosmic", f"sky_{split}.npy")

        self.data = np.load(data_path, mmap_mode="r")
        self.n = self.data.shape[0]
        self.sky = np.load(sky_path, mmap_mode="r")

        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

        self.transform = cosmic_transform()

    def __len__(self):
        return self.n

    def get_skyaug(self, i):
        """
        Return the amount of background flux to be added to image
        """
        return np.array(self.sky[i]) * np.random.uniform(
            self.skyaug_min, self.skyaug_max
        )

    def __getitem__(self, i):
        obs = np.array(self.data[i])

        image = obs[0]
        mask = obs[1]
        if obs.shape[0] == 3:
            ignore = obs[2]
        else:
            ignore = np.zeros_like(obs[0])
        # try:#JK
        skyaug = self.get_skyaug(i)

        # crop to 128*128
        image, mask, ignore = get_fixed_crop([image, mask, ignore], 128, 128)

        transformed = self.transform(np.stack([image + skyaug, mask, ignore], axis=-1))

        image_aug = transformed[:, 0:1, :, :]
        mask = torch.squeeze(transformed[:, 1:2, :, :])
        ignore = torch.squeeze(transformed[:, 2:3, :, :])

        # return image_aug, mask, ignore
        return image_aug, mask * (1.0 - ignore)


def get_random_crop(images, crop_height, crop_width):

    max_x = images[0].shape[1] - crop_width
    max_y = images[0].shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crops = []
    for image in images:
        crop = image[y : y + crop_height, x : x + crop_width]
        crops.append(crop)

    return crops


def get_fixed_crop(images, crop_height, crop_width):

    x = 64
    y = 64

    crops = []
    for image in images:
        crop = image[y : y + crop_height, x : x + crop_width]
        crops.append(crop)

    return crops


# this only changes the memory format, doesn't actually rearrange the axes
class ToChannelsLast:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError
        return x.to(memory_format=torch.channels_last)

    def __repr__(self):
        return self.__class__.__name__ + "()"


def cosmic_transform():
    def channels_to_last(img: torch.Tensor):
        return img.permute(0, 2, 3, 1).contiguous()

    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(ToChannelsLast())

    return transforms.Compose(transform_list)


def load_cosmic_data(root, val_prop=0.1, seed=1):

    aug_sky = (-0.9, 3)

    trainvalset_full = PairedDatasetImagePath(
        root=root,
        skyaug_min=aug_sky[0],
        skyaug_max=aug_sky[1],
        split="train",
        seed=seed,
    )

    testset = PairedDatasetImagePath(
        root=root, skyaug_min=aug_sky[0], skyaug_max=aug_sky[1], split="test", seed=seed
    )

    if val_prop > 0:
        train_val_size = len(trainvalset_full)
        val_size = int(train_val_size * val_prop)

        trainset = Subset(trainvalset_full, np.arange(train_val_size)[:-val_size])
        valset = Subset(trainvalset_full, np.arange(train_val_size)[-val_size:])
    else:
        valset = None

    return trainset, valset, testset


"""
ECG
"""


class ECGDataset(Dataset):
    def __init__(self, root, split="train"):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.root = root
        self.split = split

        self.x = np.load(
            os.path.join(root, "processed_data", "ecg", f"x_{split}.npy"), mmap_mode="r"
        )
        self.y = np.load(os.path.join(root, "processed_data", "ecg", f"y_{split}.npy"))
        self.y = OHE(self.y, 4)

    def __getitem__(self, index):
        x = torch.tensor(
            self.x[index].transpose(1, 0)[None, :, :, None], dtype=torch.float
        )
        y = torch.tensor(self.y[index], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.y.shape[0]


"""
DeepSEA
"""


class DeepSEA(Dataset):
    def __init__(self, root, split="train"):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.root = root
        self.split = split

        self.x = np.load(
            os.path.join(root, "processed_data", "deepsea", f"x_{split}.npy")
        )
        self.y = np.load(
            os.path.join(root, "processed_data", "deepsea", f"y_{split}.npy")
        )

    def __getitem__(self, index):
        x = torch.tensor(self.x[index][None, :, :, None], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.x.shape[0]


"""
Nottingham
"""


class Nottingham(Dataset):
    def __init__(
        self,
        root,
        split,
        pad=True,
    ):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.root = root
        self.split = split

        self.pad = pad
        self.data = np.load(
            os.path.join(root, "processed_data", "nottingham", f"{split}.npy"),
            allow_pickle=True,
        )
        self.max_len = 1793

    def __getitem__(self, index):
        x = self.data[index]
        if self.pad:
            pad_amount = self.max_len - x.shape[0]
            x = np.pad(x.astype("int"), ((pad_amount, 0), (0, 0)), constant_values=-1)

            y = torch.squeeze(torch.tensor(x[-1:, :], dtype=torch.int))
        x = torch.tensor(x[:-1, :, None, None], dtype=torch.int)

        return x, y

    def __len__(self):
        return self.data.shape[0]


"""
Navier Stokes
"""


class NavierStokes(Dataset):
    def __init__(self, root, split):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.root = root
        self.split = split

        self.x = np.load(
            os.path.join(root, "processed_data", "navierstokes", f"x_{split}.npy")
        )
        self.y = np.load(
            os.path.join(root, "processed_data", "navierstokes", f"y_{split}.npy")
        )

    def __getitem__(self, index):
        x = torch.tensor(self.x[index].transpose(2, 0, 1)[:, None, :, :])
        y = torch.tensor(self.y[index])

        return x, y

    def __len__(self):
        return self.x.shape[0]


"""
EMBER
"""


class Ember(Dataset):
    def __init__(
        self,
        root,
        split,
    ):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.root = root
        self.split = split

        self.x = np.load(
            os.path.join(root, "processed_data", "ember", f"x_{split}.npy"),
            mmap_mode="r",
        )
        self.y = np.load(
            os.path.join(root, "processed_data", "ember", f"y_{split}.npy"),
            mmap_mode="r",
        )

        self.y = OHE(self.y.astype(int), 2)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index][None, :, None, None])
        y = torch.tensor(self.y[index])

        return x, y

    def __len__(self):
        return self.x.shape[0]


"""
Crypto
"""


class Crypto(Dataset):
    def __init__(
        self,
        root,
        split="train",
    ):

        assert (
            split == "train" or split == "test"
        ), "split should be one of 'train' or 'test'"

        self.x = np.load(
            os.path.join(root, "processed_data", "crypto", f"x_{split}.npy")
        ).transpose(0, 2, 1)
        self.y = np.load(
            os.path.join(root, "processed_data", "crypto", f"y_{split}.npy")
        )

    def __getitem__(self, index):
        x = torch.tensor(self.x[index][:, :, None, None])
        y = torch.tensor(self.y[index])

        return x, y

    def __len__(self):
        return self.y.shape[0]
