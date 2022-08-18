import torch
import pickle
import os

class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, class_index):
        with open(data_path, "rb") as fh:
            self.dataset = torch.tensor(pickle.load(fh)).float()
            self.class_index = torch.tensor(class_index).float()

    def get_dataset(self):
        # for compatibility
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.class_index


def load_datasets_processed(cfg, datasets, dataset_dir=None):
    """
    load preprocessed datasets from a list, return train/test datasets, dataset index and dataset name
    """
    if dataset_dir is None:
        dataset_dir = cfg["proc_dataset_dir"]
    dataset_list = []
    class_index = 0

    for dataset_name in datasets:
        dataset_train_path = os.path.join(dataset_dir, dataset_name + "_train")
        dataset_test_path = os.path.join(dataset_dir, dataset_name + "_test")

        try:
            dataset_train = ProcessedDataset(dataset_train_path, class_index)
            dataset_test = ProcessedDataset(dataset_test_path, class_index)
        except Exception as e:
            print(e)
            continue

        dataset_list.append((dataset_train, dataset_test, dataset_name, class_index))
        class_index += 1

    return dataset_list
