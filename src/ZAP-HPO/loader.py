#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import pickle
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

_list_of_metafeatures = ['num_channels', 'num_classes', 'num_train', 'resolution_0']
_bool_hps = ["first_simple_model", "amsgrad", "nesterov"]
_categorical_hps = ['simple_model_LR', 'simple_model_NuSVC', 'simple_model_RF',
       'simple_model_SVC', 'architecture_ResNet18',
       'architecture_efficientnetb0', 'architecture_efficientnetb1',
       'architecture_efficientnetb2', 'scheduler_cosine', 'scheduler_plateau', 
       'optimiser_sgd', 'optimiser_adam', 'optimiser_adamw']
_numerical_hps = ['early_epoch', 'max_inner_loop_ratio', 'min_lr',
       'skip_valid_score_threshold', 'test_after_at_least_seconds',
       'test_after_at_least_seconds_max', 'test_after_at_least_seconds_step',
       'batch_size', 'cv_valid_ratio', 'max_size', 'max_valid_count',
       'steps_per_epoch', 'train_info_sample', 'freeze_portion',
       'lr', 'momentum', 'warm_up_epoch', 'warmup_multiplier',
       'wd']
_apply_log = ["lr","wd","min_lr"]


class TestDatabase(Dataset):
  def __init__(self, data_path, loo, input_scaler = None, output_scaler = None, use_meta=True, num_aug = 15, num_pipelines = 525):
    
    # read data
    data = pd.read_csv(os.path.join(data_path, "data_m.csv"), header=0)

    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        self.cls = pickle.load(f)
    
    test_datasets = [f"{i}-{self.cls[loo]}" for i in range(num_aug)]

    print(f"Testing on {num_aug} augmentations of {self.cls[loo]}")

    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.test_datasets = test_datasets

    X_test = np.array(attributes[data.dataset.isin(test_datasets)])
    y_test = data[data.dataset.isin(test_datasets)]["accuracy"].ravel()
    
    if input_scaler:
        len_nonorm = len(_bool_hps+_categorical_hps)
        # Scaling is done only for numerical HPs 
        X_test_transformed = input_scaler.transform(X_test[:,:-len_nonorm])
        X_test = np.concatenate([X_test_transformed, X_test[:, -len_nonorm:]], axis = 1)
    if output_scaler:
        y_test = output_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

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

class TrainDatabase(Dataset):
    
  def __getitem__(self, index):
       if self.mode == "regression":
           return self.__get_regression_item__(index)
       elif self.mode == "bpr" or self.mode == "tml":
           return self.__get_bpr_item__(index)

  def __len__(self):
    return len(self.y[self.training])

  def __get_regression_item__(self, index):
    x = self.x[self.training][index]
    y = self.y[self.training][index] 
    ystar  = self.y_star[self.training][index]

    return x, y, ystar

  def __get_bpr_item__(self, index):
    x = self.x[self.training][index]
    y = self.y[self.training][index]
    r = self.ranks_flat[self.training][index]
    
    try:
        larger_idx  = self.rng.choice(self.larger_set[index])
    except ValueError:
        larger_idx=index

    try:
        smaller_idx = self.rng.choice(self.smaller_set[index])
    except ValueError:
        smaller_idx = index

    s = self.x[self.training][smaller_idx]
    r_s = self.ranks_flat[self.training][smaller_idx]
    l = self.x[self.training][larger_idx]
    r_l = self.ranks_flat[self.training][larger_idx]
    
    return (x,s,l), (y, self.y[self.training][smaller_idx], self.y[self.training][larger_idx]), (r,r_s,r_l)

def setup_datasets(trainDB_obj, split_type='cv'):
    ''' Reads the data from data_m.csv and sets up the X, Y and rank datasets for training and validation.
    '''
    # read data
    data = pd.read_csv(os.path.join(trainDB_obj.data_path, "data_m.csv"), header=0)
    cv_folds = pd.read_csv(os.path.join(trainDB_obj.data_path, "cv_folds.csv"), header=0, index_col=0)

    if split_type=='loo':

        with open(os.path.join(trainDB_obj.data_path, "cls_names.pkl"), "rb") as f:
            trainDB_obj.cls = pickle.load(f)

        # get training/test split
        core_dataset_name = trainDB_obj.cls[trainDB_obj.loo]
        test_datasets = [f"{aug}-{core_dataset_name}" for aug in range(trainDB_obj.num_aug)]
        valid_datasets = np.setdiff1d(list(cv_folds[cv_folds["fold"].isin([trainDB_obj.cv])].index), test_datasets).tolist()
        training_datasets = np.setdiff1d(list(cv_folds[~cv_folds["fold"].isin([trainDB_obj.cv])].index), test_datasets).tolist()
    else:
        valid_datasets = list(cv_folds[cv_folds["fold"].isin([trainDB_obj.cv])].index)
        training_datasets = list(cv_folds[~cv_folds["fold"].isin([trainDB_obj.cv])].index)

    # process input
    predictors = _list_of_metafeatures + _numerical_hps + _bool_hps + _categorical_hps if trainDB_obj.use_meta else _numerical_hps + _bool_hps + _categorical_hps
    attributes = data[predictors].copy()

    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    trainDB_obj.attributes = attributes

    X_train = np.array(attributes[data.dataset.isin(training_datasets)])
    X_valid = np.array(attributes[data.dataset.isin(valid_datasets)])

    # process output
    y_train = data[data.dataset.isin(training_datasets)]["accuracy"].ravel()
    y_valid = data[data.dataset.isin(valid_datasets)]["accuracy"].ravel()
    rank_train = data[data.dataset.isin(training_datasets)]["ranks"].ravel()
    rank_valid = data[data.dataset.isin(valid_datasets)]["ranks"].ravel()

    print("Loaded and processed the meta-dataset!")

    return (X_train, X_valid, y_train, y_valid, rank_train, rank_valid)

def setup_sparsity(trainDB_obj, X_train, y_train):
    # depending on sparsity value, picks that much % from X_train along axis 0.
    # Default would be whole array size X_train.shape[0] for 0 sparsity
    # Value is then sorted and set
    if trainDB_obj.sparsity > 0:
        trainDB_obj.dense_idx = trainDB_obj.rng2.choice(trainDB_obj.ndatasets["train"], int((1 - trainDB_obj.sparsity) * trainDB_obj.ndatasets["train"]), replace=False)
        trainDB_obj.dense_idx.sort()
        X_train = X_train[trainDB_obj.dense_idx]
        y_train = y_train[trainDB_obj.dense_idx]
    
    return X_train, y_train

def setup_normalization(trainDB_obj, X_train, X_valid, y_train, y_valid):
    if trainDB_obj.input_normalization:
        len_nonorm = len(_bool_hps + _categorical_hps)
        trainDB_obj.input_scaler = StandardScaler()
        trainDB_obj.input_scaler.fit(X_train[:, :-len_nonorm])

        # Scaling is done only for numerical HPs
        X_train_transformed = trainDB_obj.input_scaler.transform(X_train[:, :-len_nonorm])
        X_train = np.concatenate([X_train_transformed, X_train[:,-len_nonorm:]], axis=1)
        
        X_valid_transformed = trainDB_obj.input_scaler.transform(X_valid[:, :-len_nonorm])
        X_valid = np.concatenate([X_valid_transformed, X_valid[:,-len_nonorm:]], axis=1)

    else:
        trainDB_obj.input_scaler = None

    if trainDB_obj.output_normalization:
        
        trainDB_obj.output_scaler = StandardScaler() 
        trainDB_obj.output_scaler.fit(y_train.reshape(-1, 1))
        
        y_train = trainDB_obj.output_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_valid = trainDB_obj.output_scaler.transform(y_valid.reshape(-1, 1)).reshape(-1)

    else:
        trainDB_obj.output_scaler = None

    return (X_train, X_valid, y_train, y_valid)

def get_y_star(trainDB_obj):
    y_star_dict = dict()
    for _set in ["train", "valid"]:
        y_star = []
        for i in range(trainDB_obj.ndatasets[_set]):
            y_star += [max(trainDB_obj.values[_set][i]) * torch.ones(trainDB_obj.num_pipelines)]
        y_star_dict[_set] = torch.cat(y_star)
    return y_star_dict

def setup_targets(trainDB_obj, y_train):
    if trainDB_obj.sparsity > 0:
        trainDB_obj.ds_ref = trainDB_obj.ds_ref[trainDB_obj.dense_idx]

        values_sparse = []
        y_star_sparse = []
        ranks_sparse = []
        ranks_flat_sparse = []
        for ds in np.unique(trainDB_obj.ds_ref):
            ds_y = y_train[np.where(trainDB_obj.ds_ref == ds)[0]]
            values_sparse.append(ds_y)
            y_star_sparse += [max(values_sparse[-1]) * torch.ones(len(values_sparse[-1]))]
            order = list(np.sort(np.unique(ds_y))[::-1])
            new_ranks = list(map(lambda x: order.index(x), ds_y))
            ranks_sparse.append(new_ranks)
            ranks_flat_sparse += (np.array(new_ranks) / max(new_ranks)).tolist()
        trainDB_obj.y_star.update({"train": torch.cat(y_star_sparse)})
        trainDB_obj.values.update({"train": values_sparse})
        trainDB_obj.ranks.update({"train": ranks_sparse})
        trainDB_obj.ranks_flat.update({"train": ranks_flat_sparse})

def setup_mode(trainDB_obj, y_train):
    if trainDB_obj.mode == "bpr" or trainDB_obj.mode == "tml":
        print(f"Setting up better/worse pipelines for the {trainDB_obj.mode} objective. Might take few minutes!")
        trainDB_obj.larger_set = []
        trainDB_obj.smaller_set = []
        for d, k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train > k, trainDB_obj.ds_ref == trainDB_obj.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train < k, trainDB_obj.ds_ref == trainDB_obj.ds_ref[d]))[0]
            trainDB_obj.larger_set.append(ll)
            trainDB_obj.smaller_set.append(ss)

class TrainDatabaseCV(TrainDatabase):
    def __init__(self, seed, data_path, cv, mode = "bpr", sparsity = 0., use_meta = True, num_pipelines = 525, input_normalization = True, output_normalization = False):
        super(TrainDatabase, self).__init__()
        self.training = "train"
        self.seed = seed
        self.data_path = data_path
        self.cv = cv
        self.mode = mode
        self.sparsity = sparsity
        self.use_meta = use_meta
        self.num_pipelines = num_pipelines
        self.output_normalization = output_normalization
        self.input_normalization = input_normalization

        self.rng = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed)
        self.valid_rng = np.random.default_rng(seed)

        X_train, X_valid, y_train, y_valid, rank_train, rank_valid = setup_datasets(self, "cv")

        self.ndatasets  = {"train":X_train.shape[0]//num_pipelines,
                           "valid":X_valid.shape[0]//num_pipelines}
        self.values     = {"train":y_train.reshape(-1,num_pipelines),
                           "valid":y_valid.reshape(-1,num_pipelines)}
        self.ranks      = {"train":rank_train.reshape(-1,num_pipelines),
                           "valid":rank_valid.reshape(-1,num_pipelines)}
        self.ranks_flat = {"train":rank_train/max(rank_train),
                           "valid":rank_valid/max(rank_valid)}
        self.y_star     = get_y_star(self)

        self.ds_ref = np.concatenate([num_pipelines*[i] for i in range(self.ndatasets["train"])])

        X_train, y_train = setup_sparsity(self, X_train, y_train)
        X_train, X_valid, y_train, y_valid = setup_normalization(self, X_train, X_valid, y_train, y_valid)
        setup_targets(self, y_train)
        setup_mode(self, y_train)

        self.x = {"train":torch.tensor(X_train.astype(np.float32)),
                  "valid":torch.tensor(X_valid.astype(np.float32))}
        self.y = {"train":torch.tensor(y_train.astype(np.float32)),
                  "valid":torch.tensor(y_valid.astype(np.float32))}
        

class TrainDatabaseCVPlusLoo(TrainDatabase):
    def __init__(self, seed, data_path, cv, loo, mode = "bpr", sparsity = 0., use_meta = True, num_aug = 15, num_pipelines = 525, input_normalization = True, output_normalization = False):
        super(TrainDatabase, self).__init__()
        self.training = "train"
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
        
        self.rng = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed)
        self.valid_rng = np.random.default_rng(seed)

        X_train, X_valid, y_train, y_valid, rank_train, rank_valid = setup_datasets(self, "loo")

        self.ndatasets  = {"train":X_train.shape[0]//num_pipelines,
                           "valid":X_valid.shape[0]//num_pipelines}
        self.values     = {"train":y_train.reshape(-1,num_pipelines),
                           "valid":y_valid.reshape(-1,num_pipelines)}
        self.ranks      = {"train":rank_train.reshape(-1,num_pipelines),
                           "valid":rank_valid.reshape(-1,num_pipelines)}
        self.ranks_flat = {"train":rank_train/max(rank_train),
                           "valid":rank_valid/max(rank_valid)}
        self.y_star     = {"train":[],
                           "valid":[]}

        self.ds_ref = np.concatenate([num_pipelines*[i] for i in range(self.ndatasets["train"])])

        X_train, y_train = setup_sparsity(self, X_train, y_train)

        X_train, X_valid, y_train, y_valid = setup_normalization(self, X_train, X_valid, y_train, y_valid)

        setup_targets(self, y_train)
        setup_mode(self, y_train)

        self.x = {"train":torch.tensor(X_train.astype(np.float32)),
                  "valid":torch.tensor(X_valid.astype(np.float32))}
        self.y = {"train":torch.tensor(y_train.astype(np.float32)),
                  "valid":torch.tensor(y_valid.astype(np.float32))}

def get_tr_loader(seed, data_path, mode = "bpr", split_type="cv", cv = 1, loo = None, sparsity = 0, use_meta=True, num_aug = 15, num_pipelines = 525, batch_size = 64):

    if split_type=="cv":
        dataset = TrainDatabaseCV(seed, data_path, cv, mode, sparsity, use_meta, num_pipelines)
    elif split_type=="loo":
        dataset = TrainDatabaseCVPlusLoo(seed, data_path, cv, loo, mode, sparsity, use_meta, num_aug, num_pipelines)
    else:
        print("Please provide a valid split type {cv|loo}")

    loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    unshuffled_loader = DataLoader(dataset = copy.deepcopy(dataset), batch_size = num_pipelines, shuffle = False)
    unshuffled_loader.dataset.mode="regression"

    print("Data setup done!")

    return loader, unshuffled_loader

def get_ts_loader(data_path, loo, input_scaler = None, output_scaler = None, use_meta=True, num_aug = 15, num_pipelines = 525):
    dataset = TestDatabase(data_path, loo, input_scaler, output_scaler, use_meta, num_aug, num_pipelines)
    loader = DataLoader(dataset = dataset, batch_size = num_pipelines, shuffle = False)
    return loader    

