#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import pickle
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

list_of_metafeatures = ['num_channels', 'num_classes', 'num_train', 'resolution_0']
bool_hps = ["first_simple_model", "amsgrad", "nesterov"]
categorical_hps = ['simple_model_LR', 'simple_model_NuSVC', 'simple_model_RF',
       'simple_model_SVC', 'architecture_ResNet18',
       'architecture_efficientnetb0', 'architecture_efficientnetb1',
       'architecture_efficientnetb2', 'scheduler_cosine', 'scheduler_plateau', 
       'optimiser_sgd', 'optimiser_adam', 'optimiser_adamw']
numerical_hps = ['early_epoch', 'max_inner_loop_ratio', 'min_lr',
       'skip_valid_score_threshold', 'test_after_at_least_seconds',
       'test_after_at_least_seconds_max', 'test_after_at_least_seconds_step',
       'batch_size', 'cv_valid_ratio', 'max_size', 'max_valid_count',
       'steps_per_epoch', 'train_info_sample', 'freeze_portion',
       'lr', 'momentum', 'warm_up_epoch', 'warmup_multiplier',
       'wd']
apply_log = ["lr","wd","min_lr"]

def normalize_input(X, input_scaler = None):
    '''
    Fits a scaler to X, transforms and returns X
    If a scaler is given only transforms and returns X
    Only fits to numerical HPs
    X: 2D array (n_datasets*n_pipelines, n_features)
    '''
    len_nonorm = len(bool_hps+categorical_hps)
    if not input_scaler:
        input_scaler = StandardScaler()
        input_scaler.fit(X[:, :-len_nonorm]) # Only fit to numerical HPs

    X_transformed = input_scaler.transform(X[:, :-len_nonorm])
    X = np.concatenate([X_transformed, X[:,-len_nonorm:]], axis=1)

    return X, input_scaler

def normalize_output(y, output_scaler = None):
    '''
    Fits a scaler to y, transforms and returns y
    If a scaler is given only transforms and returns y
    y: 1D array of length n_datasets*n_pipelines
    '''
    if not output_scaler:
        output_scaler = StandardScaler() 
        output_scaler.fit(y.reshape(-1, 1))

    y = output_scaler.transform(y.reshape(-1, 1)).reshape(-1)

    return y, output_scaler

def get_y_star(values):
    '''
    Finds the best response value per dataset
    Fills a tensor of the same length as flattened values
    values: 2D array (n_datasets, n_pipelines)
    '''
    y_star = []
    n_datasets = len(values)
    for dataset_idx in range(n_datasets):
        dataset_values = values[dataset_idx]
        n_pipelines = len(dataset_values)
        y_star += [max(dataset_values) * torch.ones(n_pipelines)]
    return torch.cat(y_star)

def get_dense_index(sparsity, sample_range, seed = 0):
    '''
    Samples random indices as many as 1-sparsity_ratio
    sparsity: Scalar between [0.0, 1)
    sample_range: Max value to sample from 
    '''
    rng = np.random.default_rng(seed)
    dense_idx = rng.choice(sample_range, int((1 - sparsity) * sample_range), replace=False)
    dense_idx.sort()
    
    return dense_idx


class TestDatabase(Dataset):
  def __init__(self, data_path, loo, input_scaler = None, output_scaler = None, use_meta=True, num_aug = 15, num_pipelines = 525):
    
    # read data
    data = pd.read_csv(os.path.join(data_path, "data_m.csv"), header=0)

    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        _cls = pickle.load(f)

    test_datasets = [f"{i}-{_cls[loo]}" for i in range(num_aug)]
    self.test_datasets = test_datasets
    print(f"Testing on {num_aug} augmentations of {_cls[loo]}")

    # Process input
    features  = list_of_metafeatures+numerical_hps+bool_hps+categorical_hps if use_meta else numerical_hps+bool_hps+categorical_hps
    X = data[features].copy()
    for feat in apply_log:
        X[feat] = X[feat].apply(lambda x: np.log(x))
    
    X_test = np.array(X[data.dataset.isin(test_datasets)])
    y_test = data[data.dataset.isin(test_datasets)]["accuracy"].ravel()

    if input_scaler:
        X_test, _ = normalize_input(X_test, input_scaler)
    if output_scaler:
        y_test, _ = normalize_input(y_test, output_scaler)

    self.x = torch.tensor(X_test.astype(np.float32))
    self.y = torch.tensor(y_test.astype(np.float32))
    self.ranks = data[data.dataset.isin(test_datasets)]["ranks"].ravel().reshape(-1, num_pipelines)
    self.values = y_test.reshape(-1, num_pipelines)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

class PredictionDatabase(Dataset):
    def __init__(self, data, input_scaler = None, use_meta = True):
        
        features = list_of_metafeatures+numerical_hps+bool_hps+categorical_hps if use_meta else numerical_hps+bool_hps+categorical_hps
        X_test = data[features].copy()
        for feat in apply_log:
            X_test[feat] = X_test[feat].apply(lambda x: np.log(x))
        X_test = np.array(X_test)
        
        if input_scaler:
            X_test, _ = normalize_input(X_test, input_scaler)

        self.x = torch.tensor(X_test.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x

class TrainDatabase(Dataset):
    
    def __getitem__(self, index):
        if self.mode == "regression":
            return self.__get_regression_item__(index)
        elif self.mode == "bpr" or self.mode == "tml":
            return self.__get_bpr_item__(index)

    def __len__(self):
        return len(self.y[self.set])

    def __get_regression_item__(self, index):
        x = self.x[self.set][index]
        y = self.y[self.set][index] 
        ystar  = self.y_star[self.set][index]

        return x, y, ystar

    def __get_bpr_item__(self, index):
        x = self.x[self.set][index]
        y = self.y[self.set][index]
        r = self.ranks_flat[self.set][index]

        try:
            larger_idx  = self.rng.choice(self.larger_set[index])
        except ValueError:
            larger_idx=index

        try:
            smaller_idx = self.rng.choice(self.smaller_set[index])
        except ValueError:
            smaller_idx = index

        s = self.x[self.set][smaller_idx]
        r_s = self.ranks_flat[self.set][smaller_idx]
        l = self.x[self.set][larger_idx]
        r_l = self.ranks_flat[self.set][larger_idx]

        return (x,s,l), (y, self.y[self.set][smaller_idx], self.y[self.set][larger_idx]), (r,r_s,r_l)

    def initialize(self, split_type):
        self.rng = np.random.default_rng(self.seed)

        X_train, X_valid, y_train, y_valid, rank_train, rank_valid = self.load_n_split_datasets(split_type)
        print("Loaded and processed the meta-dataset!")

        self.n_datasets_train  = X_train.shape[0]//self.num_pipelines
        self.n_datasets_val    = X_valid.shape[0]//self.num_pipelines
        # Reference index for datasets to work with sparse sets
        self.ds_ref = np.concatenate([self.num_pipelines*[i] for i in range(self.n_datasets_train)]) 

        dense_idx = get_dense_index(self.sparsity, X_train.shape[0], self.seed)
        X_train = X_train[dense_idx]
        y_train = y_train[dense_idx]
        self.ds_ref = self.ds_ref[dense_idx]

        self.input_scaler = None
        self.output_scaler = None

        if self.input_normalization:
            X_train, self.input_scaler = normalize_input(X_train)
            X_valid, _ = normalize_input(X_valid, self.input_scaler)

        if self.output_normalization:
            y_train, self.output_scaler = normalize_output(y_train)
            y_valid, _ = normalize_output(y_valid, self.output_scaler)

        self.set_targets(y_train, y_valid, rank_valid)
        
        if self.mode == "bpr" or self.mode == "tml":
            print(f"Setting up better/worse pipelines for the {self.mode} objective. Might take few minutes!")
            self.set_pairwise_sampling_sets(y_train)

        self.x = {"train":torch.tensor(X_train.astype(np.float32)),
                  "valid":torch.tensor(X_valid.astype(np.float32))}
        self.y = {"train":torch.tensor(y_train.astype(np.float32)),
                  "valid":torch.tensor(y_valid.astype(np.float32))}


    def load_n_split_datasets(self, split_type):
        '''
        Loads the meta-dataset, splits it according to given inner(cv) fold and outer(loo) fold
        '''
        data = pd.read_csv(os.path.join(self.data_path, "data_m.csv"), header=0)
        cv_folds = pd.read_csv(os.path.join(self.data_path, "cv_folds.csv"), header=0, index_col=0)

        # Get training/validation/test split IDs
        if split_type=='loo':
            with open(os.path.join(self.data_path, "cls_names.pkl"), "rb") as f:
                _cls = pickle.load(f)
            core_dataset_name = _cls[self.loo]
            print(f"Augmentations of {core_dataset_name} has been left out.")
            test_datasets = [f"{aug}-{core_dataset_name}" for aug in range(self.num_aug)]
            valid_datasets = np.setdiff1d(list(cv_folds[cv_folds["fold"].isin([self.cv])].index), test_datasets).tolist()
            training_datasets = np.setdiff1d(list(cv_folds[~cv_folds["fold"].isin([self.cv])].index), test_datasets).tolist()
        else:
            valid_datasets = list(cv_folds[cv_folds["fold"].isin([self.cv])].index)
            training_datasets = list(cv_folds[~cv_folds["fold"].isin([self.cv])].index)

        # Process input: Filter features and transform necessary fields
        features = list_of_metafeatures+numerical_hps+bool_hps+categorical_hps if self.use_meta else numerical_hps+bool_hps+categorical_hps
        X = data[features].copy()
        for feat in apply_log:
            X[feat] = X[feat].apply(lambda x: np.log(x))
        
        # Split into training and validation sets
        X_train = np.array(X[data.dataset.isin(training_datasets)])
        X_valid = np.array(X[data.dataset.isin(valid_datasets)])
        y_train = data[data.dataset.isin(training_datasets)]["accuracy"].ravel()
        y_valid = data[data.dataset.isin(valid_datasets)]["accuracy"].ravel()
        rank_train = data[data.dataset.isin(training_datasets)]["ranks"].ravel()
        rank_valid = data[data.dataset.isin(valid_datasets)]["ranks"].ravel()

        return (X_train, X_valid, y_train, y_valid, rank_train, rank_valid)


    def set_targets(self, y_train, y_valid, rank_valid):
        '''
        Sets training targets according to y_train
        Sets validation targets according to y_valid, rank_valid
        Training targets might be sparse, so we cannot set ranks etc. as easy as validation targets
        '''
        values_tr = []
        ranks_tr = []
        ranks_flat_tr = []
        for dataset_idx in range(self.n_datasets_train):
            dataset_values = y_train[np.where(self.ds_ref == dataset_idx)[0]]
            new_order = np.flip(dataset_values.argsort()) # Sort ASC, then flip
            new_ranks = new_order.argsort() # Sparsity changes rankings
            values_tr.append(dataset_values)
            ranks_tr.append(new_ranks)
            ranks_flat_tr.extend(new_ranks/max(new_ranks))

        # Create validation targets
        values_val = y_valid.reshape(self.n_datasets_val, self.num_pipelines)
        ranks_val = rank_valid.reshape(self.n_datasets_val, self.num_pipelines)
        ranks_flat_val = (rank_valid/max(rank_valid))

        # Get the best response value per dataset
        y_star_tr = get_y_star(values_tr)
        y_star_val = get_y_star(values_val)

        self.values     = {"train":values_tr,
                           "valid":values_val}
        self.ranks      = {"train":ranks_tr,
                           "valid":ranks_val}
        self.ranks_flat = {"train":ranks_flat_tr,
                           "valid":ranks_flat_val}
        self.y_star     = {"train":y_star_tr,
                           "valid":y_star_val}


    def set_pairwise_sampling_sets(self, y_train):
        '''
        For each dataset-pipeline response, finds better and worse performances (within dataset)
        These sets are used for sampling (anchor, better, worse) triplets
        '''
        self.larger_set = []
        self.smaller_set = []
        for d, k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train > k, self.ds_ref == self.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train < k, self.ds_ref == self.ds_ref[d]))[0]
            self.larger_set.append(ll)
            self.smaller_set.append(ss)


class TrainDatabaseCV(TrainDatabase):
    def __init__(self, seed, data_path, cv, mode = "bpr", sparsity = 0., use_meta = True, num_pipelines = 525, input_normalization = True, output_normalization = False):
        super(TrainDatabase, self).__init__()
        self.set = "train"
        self.seed = seed
        self.data_path = data_path
        self.cv = cv
        self.mode = mode
        self.sparsity = sparsity
        self.use_meta = use_meta
        self.num_pipelines = num_pipelines
        self.output_normalization = output_normalization
        self.input_normalization = input_normalization

        self.initialize("cv")
        

class TrainDatabaseCVPlusLoo(TrainDatabase):
    def __init__(self, seed, data_path, cv, loo, mode = "bpr", sparsity = 0., use_meta = True, num_aug = 15, num_pipelines = 525, input_normalization = True, output_normalization = False):
        super(TrainDatabase, self).__init__()
        self.set = "train"
        self.seed = seed
        self.data_path = data_path
        self.cv = cv
        self.loo = loo
        self.mode = mode
        self.sparsity = sparsity
        self.use_meta = use_meta
        self.num_aug = num_aug
        self.num_pipelines = num_pipelines
        self.output_normalization = output_normalization
        self.input_normalization = input_normalization
        
        self.initialize("loo")

def get_tr_loader(seed, data_path, mode = "bpr", split_type="cv", cv = 1, loo = None, sparsity = 0, use_meta=True, num_aug = 15, num_pipelines = 525, batch_size = 64):

    if split_type=="cv":
        dataset = TrainDatabaseCV(seed, data_path, cv, mode, sparsity, use_meta, num_pipelines)
    elif split_type=="loo":
        dataset = TrainDatabaseCVPlusLoo(seed, data_path, cv, loo, mode, sparsity, use_meta, num_aug, num_pipelines)
    else:
        print("Please provide a valid split type {cv|loo}")

    loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    # Loader for validating the meta-training procedure
    unshuffled_loader = DataLoader(dataset = copy.deepcopy(dataset), batch_size = num_pipelines, shuffle = False)
    unshuffled_loader.dataset.mode="regression" # Don't need to sample better/worse samples

    print("Data setup done!")

    return loader, unshuffled_loader

def get_ts_loader(data_path, loo, input_scaler = None, output_scaler = None, use_meta = True, num_aug = 15, num_pipelines = 525):
    dataset = TestDatabase(data_path, loo, input_scaler, output_scaler, use_meta, num_aug, num_pipelines)
    loader = DataLoader(dataset = dataset, batch_size = num_pipelines, shuffle = False)
    return loader    

def get_pred_loader(data, input_scaler = None, use_meta = True):
    dataset = PredictionDatabase(data, input_scaler, use_meta)
    loader = DataLoader(dataset = dataset, batch_size = data.shape[0], shuffle = False)
    return loader   

