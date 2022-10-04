import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset


from .fsd50kutils.audio_dataset import _collate_fn, _collate_fn_eval
from .dev_datasets import DecathlonDataset

"""
General function for returning the train, val (optional), and test dataloaders for any specified dev task
"""


def get_dev_dataloaders(
    task: str,
    root: str,
    val_prop: float,
    batch_size: int,
    num_workers: int = 0,
    seed=None,
):
    """
    task: string indicating task (see DecathlonDataset class for valid options)
    root: root directory that contains all the dev data
    val_prop: If val_prop>0, the proportion of the train set to split off as validation. Otherwise, only return a train and test set
    batch_size: passed to torch DataLoader()
    num_workers: passed to torch DataLoader()
    seed: can specify random seed in validation split for reproducibility

    """
    use_val = val_prop > 0

    train_set = DecathlonDataset(task=task, root=root, split="train")
    test_set = DecathlonDataset(task=task, root=root, split="test")
    val_set = None

    if use_val:  # split the training set
        if task == "fsd50k":  # fsd50k has a pre-split validation set
            val_set = DecathlonDataset(task=task, root=root, split="val")
        else:  # subset the train set
            train_size = len(train_set)
            val_size = int(train_size * val_prop)
            train_size = train_size - val_size

            if seed:
                train_set, val_set = torch.utils.data.random_split(
                    train_set,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed),
                )
            else:
                train_set, val_set = torch.utils.data.random_split(
                    train_set, [train_size, val_size]
                )

    # creating loaders
    if task == "fsd50k":
        fsd50k_batch_size = 64
        # fsd50k also needs custom collate functions
        train_loader = DataLoader(
            train_set,
            batch_size=fsd50k_batch_size,
            num_workers=num_workers,
            collate_fn=_collate_fn,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=fsd50k_batch_size,
            num_workers=num_workers,
            collate_fn=_collate_fn_eval,
        )
        val_loader = (
            DataLoader(
                val_set,
                batch_size=fsd50k_batch_size,
                num_workers=num_workers,
                collate_fn=_collate_fn_eval,
            )
            if val_set
            else None
        )

    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = (
            DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
            if val_set
            else None
        )

    return train_loader, val_loader, test_loader
