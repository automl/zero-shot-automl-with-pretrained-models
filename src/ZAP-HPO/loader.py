#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:59:03 2021

@author: hsjomaa
"""
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import pickle
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

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
    data =pd.read_csv(os.path.join(data_path, "data_m.csv"), header=0)

    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        self.cls = pickle.load(f)
    
    testing_cls = [f"{i}-{self.cls[loo]}" for i in range(num_aug)]

    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.testing_cls = testing_cls

    X_test = np.array(attributes[data.dataset.isin(testing_cls)])
    y_test = data[data.dataset.isin(testing_cls)]["accuracy"].ravel()
    
    if input_scaler:
        len_bool_cat = len(_bool_hps+_categorical_hps)
        # Scaling is done only for numerical HPs 
        X_test = np.concatenate([input_scaler.transform(X_test[:,:-len_bool_cat]), X_test[:, X_test.shape[1]-len_bool_cat:]], axis = 1)
    if output_scaler:
        y_test = output_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    self.x = torch.tensor(X_test.astype(np.float32)) # this could be torch.from_numpy -> maybe faster
    self.y = torch.tensor(y_test.astype(np.float32))
    self.ranks = data[data.dataset.isin(testing_cls)]["ranks"].ravel().reshape(-1, num_pipelines)
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
    
class TrainDatabaseCV(TrainDatabase):
  def __init__(self, seed, data_path, cv, output_normalization=False, input_normalization=True, mode="regression", sparsity = 0., use_meta=True, num_pipelines = 525):
    super(TrainDatabase, self).__init__()
    self.training = "train"
    self.output_normalization = output_normalization
    self.input_normalization = input_normalization
    self.mode = mode
    self.sparsity = sparsity
    self.rng = np.random.default_rng(seed)
    self.rng2 = np.random.default_rng(seed)
    self.valid_rng = np.random.default_rng(seed)

    # read data
    data = pd.read_csv(os.path.join(data_path,"data_m.csv"),header=0)
    cv_folds = pd.read_csv(os.path.join(data_path,"cv_folds.csv"), header=0, index_col = 0)

    valid_cls = list(cv_folds[cv_folds["fold"].isin([cv])].index)
    training_cls = list(cv_folds[~cv_folds["fold"].isin([cv])].index)

    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()

    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.attributes = attributes
    
    X_train = np.array(attributes[data.dataset.isin(training_cls)])
    X_valid = np.array(attributes[data.dataset.isin(valid_cls)])
    
    if self.sparsity>0:
        dense_idx = self.rng2.choice(X_train.shape[0],int((1-self.sparsity)*X_train.shape[0]),replace=False)
        dense_idx.sort()
        self.dense_idx = dense_idx
        X_train = X_train[dense_idx]
    
    self.ndatasets = {"train": X_train.shape[0]//num_pipelines, "valid": X_valid.shape[0]//num_pipelines}

    if input_normalization:
        len_bool_cat = len(_bool_hps+_categorical_hps)
        
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X_train[:,:-len_bool_cat])

        # Scaling is done only for numerical HPs 
        X_train = np.concatenate([self.input_scaler.transform(X_train[:,:-len_bool_cat]), X_train[:, X_train.shape[1]-len_bool_cat:]], axis = 1)
        X_valid = np.concatenate([self.input_scaler.transform(X_valid[:,:-len_bool_cat]), X_valid[:, X_train.shape[1]-len_bool_cat:]], axis = 1)
    
    # process output
    y_train = data[data.dataset.isin(training_cls)]["accuracy"].ravel()
    y_valid= data[data.dataset.isin(valid_cls)]["accuracy"].ravel()
    rank_train = data[data.dataset.isin(training_cls)]["ranks"].ravel()
    rank_valid = data[data.dataset.isin(valid_cls)]["ranks"].ravel()

    if output_normalization:
        self.output_scaler = StandardScaler()
        self.output_scaler.fit(y_train.reshape(-1,1))

        y_train = self.output_scaler.transform(y_train.reshape(-1,1)).reshape(-1)
        y_valid = self.output_scaler.transform(y_valid.reshape(-1,1)).reshape(-1)

    self.x = {"train":torch.tensor(X_train.astype(np.float32)),
              "valid":torch.tensor(X_valid.astype(np.float32))}
    self.y = {"train":torch.tensor(y_train.astype(np.float32)),
              "valid":torch.tensor(y_valid.astype(np.float32))}

    # meta-dataset size 525 is hard-coded and should be corrected

    self.values = {"train":y_train.reshape(-1,num_pipelines),"valid":y_valid.reshape(-1,num_pipelines)}
    self.ranks = {"train":rank_train.reshape(-1,num_pipelines),"valid":rank_valid.reshape(-1,num_pipelines)}
    self.ranks_flat = dict()
    self.ranks_flat["valid"] = rank_valid/max(rank_valid)
    self.ranks_flat["train"] = rank_train/max(rank_train) 
    self.y_star = {"train":[],"valid": []}
    self.ds_ref = np.concatenate([num_pipelines*[i] for i in range(self.ndatasets["train"])])

    if self.sparsity > 0:
        self.ds_ref = self.ds_ref[dense_idx]    

        values_sparse = []
        y_star_sparse = []
        ranks_sparse = []
        ranks_flat = []
        for ds in np.unique(self.ds_ref):
            ds_y =  self.y["train"][dense_idx][np.where(self.ds_ref==ds)[0]]
            values_sparse.append(ds_y)
            y_star_sparse += [max(values_sparse[-1])*torch.ones(len(values_sparse[-1]))]
            orde      = list(np.sort(np.unique(ds_y))[::-1])
            new_ranks = list(map(lambda x: orde.index(x),ds_y))
            ranks_sparse.append(new_ranks)                
            ranks_flat+=(np.array(new_ranks)/max(new_ranks)).tolist()
        self.y_star.update({"train":torch.cat(y_star_sparse)})                
        self.values.update({"train":values_sparse})                
        self.ranks.update({"train":ranks_sparse})
        self.y.update({"train":self.y["train"][dense_idx]})
        self.ranks_flat.update({"train":ranks_flat})
    
        # values, ranks and y stay the same
        y_star = []
        for i in range(self.ndatasets["valid"]):
            y_star += [max(self.values["valid"][i])*torch.ones(num_pipelines)]
        self.y_star.update({"valid":torch.cat(y_star)})        

    else:
        for _set in ["train","valid"]:
            y_star = []
            for i in range(self.ndatasets[_set]):
                y_star += [max(self.values[_set][i])*torch.ones(num_pipelines)]
            self.y_star.update({_set:torch.cat(y_star)})

    y_train = y_train if self.sparsity == 0 else y_train[dense_idx]
    
    if mode=="bpr" or mode=="tml":
        self.larger_set = []
        self.smaller_set = []
        for d,k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train>k,self.ds_ref==self.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train<k,self.ds_ref==self.ds_ref[d]))[0]
            self.larger_set.append(ll)
            self.smaller_set.append(ss)
    
class TrainDatabaseCVPlusLoo(TrainDatabase):
  def __init__(self, seed, data_path, loo, cv, output_normalization=False, input_normalization=True, mode="regression", sparsity = 0., use_meta=True, num_aug = 15, num_pipelines = 525):
    super(TrainDatabase, self).__init__()
    self.training = "train"
    self.output_normalization = output_normalization
    self.input_normalization = input_normalization
    self.mode = mode
    self.sparsity = sparsity
    self.rng = np.random.default_rng(seed)
    self.rng2 = np.random.default_rng(seed)
    self.valid_rng = np.random.default_rng(seed)
    # read data
    data = pd.read_csv(os.path.join(data_path,"data_m.csv"),header=0)
    cv_folds = pd.read_csv(os.path.join(data_path,"cv_folds.csv"), header=0, index_col = 0)
    
    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        self.cls = pickle.load(f)
    
    # get training/test split 
    exclude_cls_original = self.cls[loo]
    exclude_cls = []
    for aug in range(num_aug):
        exclude_cls.append(f"{aug}-{exclude_cls_original}")
    valid_cls = np.setdiff1d(list(cv_folds[cv_folds["fold"].isin([cv])].index), exclude_cls).tolist()
    training_cls = np.setdiff1d(list(cv_folds[~cv_folds["fold"].isin([cv])].index), exclude_cls).tolist()
    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()
    
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.attributes = attributes

    X_train = np.array(attributes[data.dataset.isin(training_cls)])
    X_valid = np.array(attributes[data.dataset.isin(valid_cls)])
    self.ndatasets = {"train": X_train.shape[0]//num_pipelines, "valid": X_valid.shape[0]//num_pipelines}
    
    if self.sparsity>0:
        dense_idx = self.rng2.choice(X_train.shape[0],int((1-self.sparsity)*X_train.shape[0]),replace=False)
        dense_idx.sort()
        self.dense_idx = dense_idx
        X_train = X_train[dense_idx]

    if input_normalization:
        len_bool_cat = len(_bool_hps+_categorical_hps)
        
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X_train[:,:-len_bool_cat])

        # Scaling is done only for numerical HPs 
        X_train = np.concatenate([self.input_scaler.transform(X_train[:,:-len_bool_cat]), X_train[:, X_train.shape[1]-len_bool_cat:]], axis = 1)
        X_valid = np.concatenate([self.input_scaler.transform(X_valid[:,:-len_bool_cat]), X_valid[:, X_train.shape[1]-len_bool_cat:]], axis = 1)

    
    # process output
    y_train = data[data.dataset.isin(training_cls)]["accuracy"].ravel()
    y_valid= data[data.dataset.isin(valid_cls)]["accuracy"].ravel()
    rank_train = data[data.dataset.isin(training_cls)]["ranks"].ravel()
    rank_valid = data[data.dataset.isin(valid_cls)]["ranks"].ravel()

    if output_normalization:
        self.output_scaler = StandardScaler()
        self.output_scaler.fit(y_train.reshape(-1,1))

        y_train = self.output_scaler.transform(y_train.reshape(-1,1)).reshape(-1)
        y_valid = self.output_scaler.transform(y_valid.reshape(-1,1)).reshape(-1)

    self.x = {"train":torch.tensor(X_train.astype(np.float32)),
              "valid":torch.tensor(X_valid.astype(np.float32))}
    self.y = {"train":torch.tensor(y_train.astype(np.float32)),
              "valid":torch.tensor(y_valid.astype(np.float32))}
    
    self.values = {"train":y_train.reshape(-1,num_pipelines),"valid":y_valid.reshape(-1,num_pipelines)}
    
    self.ranks = {"train":rank_train.reshape(-1,num_pipelines),"valid":rank_valid.reshape(-1,num_pipelines)}
    self.ranks_flat = dict()
    self.ranks_flat["valid"] = rank_valid/max(rank_valid)
    self.ranks_flat["train"] = rank_train/max(rank_train)
    self.y_star = {"train":[],"valid": []}
    
    self.ds_ref = np.concatenate([num_pipelines*[i] for i in range(self.ndatasets["train"])])
    
    if self.sparsity > 0:
        self.ds_ref = self.ds_ref[dense_idx]    

        values_sparse = []
        y_star_sparse = []
        ranks_sparse = []
        ranks_flat = []
        for ds in np.unique(self.ds_ref):
            ds_y =  self.y["train"][dense_idx][np.where(self.ds_ref==ds)[0]]
            values_sparse.append(ds_y)
            y_star_sparse += [max(values_sparse[-1])*torch.ones(len(values_sparse[-1]))]
            orde      = list(np.sort(np.unique(ds_y))[::-1])
            new_ranks = list(map(lambda x: orde.index(x),ds_y))
            ranks_sparse.append(new_ranks)                
            ranks_flat+=(np.array(new_ranks)/max(new_ranks)).tolist()
        self.y_star.update({"train":torch.cat(y_star_sparse)})                
        self.values.update({"train":values_sparse})                
        self.ranks.update({"train":ranks_sparse})
        self.y.update({"train":self.y["train"][dense_idx]})
        self.ranks_flat.update({"train":ranks_flat})

        # values, ranks and y stay the same
        
        y_star = []
        for i in range(self.ndatasets["valid"]):
            y_star += [max(self.values["valid"][i])*torch.ones(num_pipelines)]
        self.y_star.update({"valid":torch.cat(y_star)})            
    else:
        for _set in ["train","valid"]:
            y_star = []
            for i in range(self.ndatasets[_set]):
                y_star += [max(self.values[_set][i])*torch.ones(num_pipelines)]
            self.y_star.update({_set:torch.cat(y_star)})

    y_train = y_train if self.sparsity == 0 else y_train[dense_idx]
    
    if mode=="bpr" or "tml":
        self.larger_set = []
        self.smaller_set = []
        for d,k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train>k,self.ds_ref==self.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train<k,self.ds_ref==self.ds_ref[d]))[0]
            self.larger_set.append(ll)
            self.smaller_set.append(ss)


def get_tr_loader(seed, batch_size, data_path, loo, mode, use_meta=True, cv=None, split_type="cv", sparsity = 0, output_normalization = True, num_aug = 15, num_pipelines = 525):
    
    if split_type=="cv":
        dataset = TrainDatabaseCV(seed, data_path, cv=cv, mode=mode, sparsity = sparsity, use_meta=use_meta, output_normalization = output_normalization, num_pipelines = num_pipelines)
    elif split_type=="loo":
        dataset = TrainDatabaseCVPlusLoo(seed, data_path, cv=cv, loo=loo, mode=mode, sparsity=sparsity, output_normalization = output_normalization, use_meta=use_meta, num_aug = num_aug, num_pipelines = num_pipelines)
    else:
        print("Please provide a valid split type {cv|loo}")

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    unshuffled_loader = DataLoader(dataset=copy.deepcopy(dataset), batch_size=num_pipelines, shuffle=False)
    unshuffled_loader.dataset.mode="regression"

    return loader, unshuffled_loader

def get_ts_loader(data_path, loo, input_scaler = None, output_scaler = None, use_meta=True, num_aug = 15, num_pipelines = 525):
    dataset = TestDatabase(data_path, loo, input_scaler = input_scaler, output_scaler = output_scaler, use_meta=use_meta, num_aug = num_aug, num_pipelines = num_pipelines)
    loader = DataLoader(dataset=dataset, batch_size=num_pipelines, shuffle=False)
    return loader    

