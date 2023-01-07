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
_list_of_metafeatures = ['num_channels', 'num_classes', 'num_train', 'resolution_0']
_categorical_hps = ['simple_model_LR', 'simple_model_NuSVC', 'simple_model_RF',
'simple_model_SVC', 'architecture_ResNet18',
'architecture_efficientnetb0', 'architecture_efficientnetb1',
'architecture_efficientnetb2', 'scheduler_cosine', 'scheduler_plateau', 
'optimiser_sgd', 'optimiser_adam', 'optimiser_adamw']
_bool_hps = ["first_simple_model", "amsgrad", "nesterov"]
'''
_numerical_hps = ['early_epoch', 'max_inner_loop_ratio', 'min_lr',
'skip_valid_score_threshold', 'test_after_at_least_seconds',
'test_after_at_least_seconds_max', 'test_after_at_least_seconds_step',
'batch_size', 'cv_valid_ratio', 'max_size', 'max_valid_count',
'steps_per_epoch', 'train_info_sample', 'freeze_portion',
'lr', 'momentum', 'warm_up_epoch', 'warmup_multiplier',
'wd']
'''
_numerical_hps = ['early_epoch', 'first_simple_model', 'max_inner_loop_ratio', 'min_lr',
'skip_valid_score_threshold', 'test_after_at_least_seconds',
'test_after_at_least_seconds_max', 'test_after_at_least_seconds_step',
'batch_size', 'cv_valid_ratio', 'max_size', 'max_valid_count',
'steps_per_epoch', 'train_info_sample', 'amsgrad', 'freeze_portion',
'lr', 'momentum', 'nesterov', 'warm_up_epoch', 'warmup_multiplier',
'wd']
_apply_log = ["lr","wd","min_lr"]

class TestDatabase(Dataset):
  def __init__(self, data_path, loo, mean_input, std_input, mean_output, std_output,split_type="cv", use_meta=True):
    
    # read data
    data =pd.read_csv(os.path.join(data_path,"data_m.csv"),header=0)
    if split_type=="loo":
        with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
            self.cls = pickle.load(f)
        
        testing_cls = [f"{i}-{self.cls[loo]}" for i in range(15)]
    else:
        cv_folds = pd.read_csv(os.path.join(data_path,"cv_folds.csv"),header=0)
        testing_cls = list(cv_folds[cv_folds["fold"].isin([loo])]["Unnamed: 0"])    
    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.testing_cls = testing_cls
    self.mean_input = torch.from_numpy(mean_input)
    self.std_input = torch.from_numpy(std_input)

    self.mean_output = torch.from_numpy(mean_output)
    self.std_output = torch.from_numpy(std_output)
    self.xtest = attributes[data.dataset.isin(testing_cls)]
    X_test = np.array(attributes[data.dataset.isin(testing_cls)])
    y_test = data[data.dataset.isin(testing_cls)]["accuracy"].ravel()
    
    self.x = torch.tensor(X_test.astype(np.float32))
    self.y = torch.tensor(y_test.astype(np.float32))
    self.ranks = data[data.dataset.isin(testing_cls)]["ranks"].ravel().reshape(-1,525)
    self.values = y_test.reshape(-1,525)
    
  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    x = self.x[index]
    y = self.y[index]
    y = ((y - self.mean_output) / self.std_output)
    x = ((x - self.mean_input) / self.std_input)
    return x, y

class TestDatabaseOnline(Dataset):
  def __init__(self, data, loo, mean_input, std_input, split_type="cv", use_meta=True):
       
    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    print(f"predictors {len(predictors)}")
    attributes = data[predictors].copy()
    print(f"attributes {len(attributes)}")
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.mean_input = torch.from_numpy(mean_input)
    self.std_input = torch.from_numpy(std_input)

    X_test = np.array(attributes)

    self.x = torch.tensor(X_test.astype(np.float32))
    
  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    x = self.x[index]
    x = ((x- self.mean_input) / self.std_input)
    return x


class TrainDatabase(Dataset):
    
  def __getitem__(self, index):
       if self.mode == "regression":
           return self.__get_regression_item__(index)
       else:
           return self.__get_bpr_item__(index)

  def __len__(self):
    return len(self.y[self.training])

  def __get_regression_item__(self, index):
    # ridx = self.idx_lst[self.mode]
    x = self.x[self.training][index]
    y = self.y[self.training][index]
    ystar  = self.y_star[self.training][index]
    y = ((y- self.mean_output) / self.std_output)
    ys = ((ystar- self.mean_output) / self.std_output)
    x = ((x- self.mean_input) / self.std_input)
    return x, y, ys

  def __get_bpr_item__(self, index):
    x = self.x[self.training][index]
    y = self.y[self.training][index]
    r = self.ranks_flat[self.training][index]
    # ds = self.ds_ref[index]
    try:
        # larger_idx  = self.rng.choice(np.where(np.logical_and(self.y[self.training]>y,self.ds_ref==ds))[0])
        larger_idx  = self.rng.choice(self.larger_set[index])
    except ValueError:
        larger_idx=index
    try:
        # smaller_idx = self.rng.choice(np.where(np.logical_and(self.y[self.training]<y ,self.ds_ref==ds))[0])
        smaller_idx = self.rng.choice(self.smaller_set[index])
    except ValueError:
        smaller_idx = index
    x = ((x- self.mean_input) / self.std_input)
    s = ((self.x[self.training][smaller_idx] - self.mean_input) / self.std_input)
    r_s = self.ranks_flat[self.training][smaller_idx]
    l = ((self.x[self.training][larger_idx]- self.mean_input) / self.std_input)
    r_l = self.ranks_flat[self.training][larger_idx]
    return (x,s,l), (y,self.y[self.training][smaller_idx],self.y[self.training][larger_idx]), (r,r_s,r_l)
    
class TrainDatabaseCV(TrainDatabase):
  def __init__(self, data_path, cv, output_normalization=False,
               input_normalization=True, mode="regression"):
    super(TrainDatabase, self).__init__()
    self.training = "train"
    self.output_normalization = output_normalization
    self.input_normalization = input_normalization
    self.mode = mode
    self.rng = np.random.RandomState(0)
    self.valid_rng = np.random.RandomState(0)
    # read data
    data = pd.read_csv(os.path.join(data_path,"data_m.csv"),header=0)
    cv_folds = pd.read_csv(os.path.join(data_path,"cv_folds.csv"),header=0)
    cls = list(cv_folds[~cv_folds["fold"].isin([cv])]["Unnamed: 0"].apply(lambda x: x.split("-")[1]).unique())
    valid_cls_names = self.valid_rng.choice(cls,replace=False,size=3)
    valid_cls = []
    for v in valid_cls_names:
        for i in range(15):
            valid_cls.append(f"{i}-{v}")
    training_cls = np.setdiff1d(list(cv_folds[~cv_folds["fold"].isin([cv])]["Unnamed: 0"]),valid_cls).tolist()
    # process input
    attributes = data[_list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps].copy()
    
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.attributes = attributes
    X_train = np.array(attributes[data.dataset.isin(training_cls)])
    X_valid = np.array(attributes[data.dataset.isin(valid_cls)])
    if input_normalization:
        self.std_input = np.concatenate([np.std(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                         np.ones(len(_bool_hps+_categorical_hps))]).astype(np.float32)
        self.std_input[self.std_input == 0 ] = 1
        self.mean_input = np.concatenate([np.mean(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                          np.zeros(len(_bool_hps+_categorical_hps))]).astype(np.float32)
    else:
        self.std_input = np.ones(X_train.shape[ 1 ]).astype(np.float32)
        self.mean_input = np.zeros(X_train.shape[ 1 ]).astype(np.float32)

    # process output
    y_train = data[data.dataset.isin(training_cls)]["accuracy"].ravel()
    y_valid= data[data.dataset.isin(valid_cls)]["accuracy"].ravel()
    rank_train = data[data.dataset.isin(training_cls)]["ranks"].ravel()
    rank_valid = data[data.dataset.isin(valid_cls)]["ranks"].ravel()
    if output_normalization:
        self.mean_output = np.mean(y_train).astype(np.float32)
        self.std_output = np.std(y_train).astype(np.float32)
        try:
            assert self.std_output !=0.
        except Exception:
            self.std_output = 1.
    else:
        self.mean_output = 0.
        self.std_output = 1.

    self.x = {"train":torch.tensor(X_train.astype(np.float32)),
              "valid":torch.tensor(X_valid.astype(np.float32))}
    self.y = {"train":torch.tensor(y_train.astype(np.float32)),
              "valid":torch.tensor(y_valid.astype(np.float32))}
    self.values = {"train":y_train.reshape(-1,525),"valid":y_valid.reshape(-1,525)}
    self.ranks = {"train":rank_train.reshape(-1,525),"valid":rank_valid.reshape(-1,525)}
    self.ranks_flat = dict()
    self.ranks_flat["valid"] = rank_valid/max(rank_valid)
    self.ranks_flat["train"] = rank_train/max(rank_train) 
    self.y_star = {"train":[],"valid": []}
    self.ds_ref = np.concatenate([525*[i] for i in range(len(self.y["train"])//525)])
    for thing in ["train","valid"]:
        ndatasets = len(self.y[thing])//525
        y_star = []
        for i in range(ndatasets):
            y_star += [max(self.values[thing][i])*torch.ones(525)]
        self.y_star.update({thing:torch.cat(y_star)})
    if mode=="bpr":
        self.larger_set = []
        self.smaller_set = []
        for d,k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train>k,self.ds_ref==self.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train<k,self.ds_ref==self.ds_ref[d]))[0]
            self.larger_set.append(ll)
            self.smaller_set.append(ss)
    
class TrainDatabaseCVPlusLoo(TrainDatabase):
  def __init__(self, data_path, loo, cv, output_normalization=False,
               input_normalization=True, mode="regression", sparsity = 0., use_meta=True):
    super(TrainDatabase, self).__init__()
    self.training = "train"
    self.output_normalization = output_normalization
    self.input_normalization = input_normalization
    self.mode = mode
    self.sparsity = sparsity
    self.rng = np.random.RandomState(0)
    self.rng2 = np.random.RandomState(0)
    self.valid_rng = np.random.RandomState(0)
    # read data
    data = pd.read_csv(os.path.join(data_path,"data_m.csv"),header=0)
    cv_folds = pd.read_csv(os.path.join(data_path,"cv_folds.csv"),header=0)
    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        self.cls = pickle.load(f)
    # get training/test split 
    exclude_cls_original = self.cls[loo]
    exclude_cls = []
    for aug in range(15):
        exclude_cls.append(f"{aug}-{exclude_cls_original}")
    valid_cls = np.setdiff1d(list(cv_folds[cv_folds["fold"].isin([cv])]["Unnamed: 0"]),exclude_cls).tolist()
    training_cls = np.setdiff1d(list(cv_folds[~cv_folds["fold"].isin([cv])]["Unnamed: 0"]),exclude_cls).tolist()
    # process input
    predictors  = _list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps if use_meta else _numerical_hps+_bool_hps+_categorical_hps
    attributes = data[predictors].copy()
    
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    self.attributes = attributes
    X_train = np.array(attributes[data.dataset.isin(training_cls)])
    if self.sparsity>0:
        dense_idx = self.rng2.choice(X_train.shape[0],int((1-self.sparsity)*X_train.shape[0]),replace=False)
        dense_idx.sort()
        self.dense_idx = dense_idx
        X_train = X_train[dense_idx]
    X_valid = np.array(attributes[data.dataset.isin(valid_cls)])
    if input_normalization:
        self.std_input = np.concatenate([np.std(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                         np.ones(len(_bool_hps+_categorical_hps))]).astype(np.float32)
        self.std_input[self.std_input == 0 ] = 1
        self.mean_input = np.concatenate([np.mean(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                          np.zeros(len(_bool_hps+_categorical_hps))]).astype(np.float32)
    else:
        self.std_input = np.ones(X_train.shape[ 1 ]).astype(np.float32)
        self.mean_input = np.zeros(X_train.shape[ 1 ]).astype(np.float32)

    self.x = {"train":torch.tensor(X_train.astype(np.float32)),
              "valid":torch.tensor(X_valid.astype(np.float32))}
    
    # process output
    y_train = data[data.dataset.isin(training_cls)]["accuracy"].ravel()
    y_valid= data[data.dataset.isin(valid_cls)]["accuracy"].ravel()
    rank_train = data[data.dataset.isin(training_cls)]["ranks"].ravel()
    rank_valid = data[data.dataset.isin(valid_cls)]["ranks"].ravel()
    # train will be changed in case of missing values
    self.y = {"train":torch.tensor(y_train.astype(np.float32)),
              "valid":torch.tensor(y_valid.astype(np.float32))}
    
    self.values = {"train":y_train.reshape(-1,525),"valid":y_valid.reshape(-1,525)}
    
    self.ranks = {"train":rank_train.reshape(-1,525),"valid":rank_valid.reshape(-1,525)}
    self.ranks_flat = dict()
    self.ranks_flat["valid"] = rank_valid/max(rank_valid)
    self.ranks_flat["train"] = rank_train/max(rank_train)
    self.y_star = {"train":[],"valid": []}
    
    self.ds_ref = np.concatenate([525*[i] for i in range(len(self.y["train"])//525)])
    
    if self.sparsity > 0:
        self.ds_ref = self.ds_ref[dense_idx]    
        for thing in ["train"]:
            values_sparse = []
            y_star_sparse = []
            ranks_sparse = []
            ranks_flat = []
            for ds in np.unique(self.ds_ref):
                ds_y =  self.y[thing][dense_idx][np.where(self.ds_ref==ds)[0]]
                values_sparse.append(ds_y)
                y_star_sparse += [max(values_sparse[-1])*torch.ones(len(values_sparse[-1]))]
                orde      = list(np.sort(np.unique(ds_y))[::-1])
                new_ranks = list(map(lambda x: orde.index(x),ds_y))
                ranks_sparse.append(new_ranks)                
                ranks_flat+=(np.array(new_ranks)/max(new_ranks)).tolist()
            self.y_star.update({thing:torch.cat(y_star_sparse)})                
            self.values.update({thing:values_sparse})                
            self.ranks.update({thing:ranks_sparse})
            self.y.update({thing:self.y[thing][dense_idx]})
            self.ranks_flat.update({thing:ranks_flat})
        for thing in ["valid"]:
            # values, ranks and y stay the same
            ndatasets = len(self.y[thing])//525
            y_star = []
            for i in range(ndatasets):
                y_star += [max(self.values[thing][i])*torch.ones(525)]
            self.y_star.update({thing:torch.cat(y_star)})            
    else:
        for thing in ["train","valid"]:
            ndatasets = len(self.y[thing])//525
            y_star = []
            for i in range(ndatasets):
                y_star += [max(self.values[thing][i])*torch.ones(525)]
            self.y_star.update({thing:torch.cat(y_star)})
    y_train = y_train if self.sparsity == 0 else y_train[dense_idx]
    if mode=="bpr":
        self.larger_set = []
        self.smaller_set = []
        for d,k in enumerate(y_train):
            ll = np.where(np.logical_and(y_train>k,self.ds_ref==self.ds_ref[d]))[0]
            ss = np.where(np.logical_and(y_train<k,self.ds_ref==self.ds_ref[d]))[0]
            self.larger_set.append(ll)
            self.smaller_set.append(ss)

    if output_normalization:
        self.mean_output = np.mean(y_train).astype(np.float32)
        self.std_output = np.std(y_train).astype(np.float32)
        try:
            assert self.std_output !=0.
        except Exception:
            self.std_output = 1.
    else:
        self.mean_output = 0.
        self.std_output = 1.            
            
class TrainDatabaseLoo(TrainDatabase):
  def __init__(self, data_path, loo, output_normalization=False,
               input_normalization=True, mode="regression"):
    super(TrainDatabase, self).__init__()
    self.training = "train"
    self.output_normalization = output_normalization
    self.input_normalization = input_normalization
    self.mode = mode
    self.rng = np.random.RandomState(0)
    # read data
    data =pd.read_csv(os.path.join(data_path,"data_merged.csv"),header=0)
    with open(os.path.join(data_path,"cls_names.pkl"),"rb") as f:
        self.cls = pickle.load(f)
    
    # get training/test split 
    training_cls_idx = np.setdiff1d(np.arange(len(self.cls)),loo).tolist()
    training_cls = []
    for cls_idx in training_cls_idx:
        for aug in range(15):
            training_cls.append(f"{aug}-{self.cls[cls_idx]}")
    # testing_cls = [f"0-{self.cls[loo]}"]
    
    # process input
    attributes = data[_list_of_metafeatures+_numerical_hps+_bool_hps+_categorical_hps]
    for alog in _apply_log:
        attributes[alog] = attributes[alog].apply(lambda x: np.log(x))
    
    X_train = np.array(attributes[data.dataset.isin(training_cls)])
    if input_normalization:
        self.std_input = np.concatenate([np.std(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                         np.ones(len(_bool_hps+_categorical_hps))]).astype(np.float32)
        self.std_input[self.std_input == 0 ] = 1
        self.mean_input = np.concatenate([np.mean(X_train[:,:-len(_bool_hps+_categorical_hps)], 0),
                                          np.zeros(len(_bool_hps+_categorical_hps))]).astype(np.float32)
    else:
        self.std_input = np.ones(X_train.shape[ 1 ]).astype(np.float32)
        self.mean_input = np.zeros(X_train.shape[ 1 ]).astype(np.float32)

    # process output
    y_train = data[data.dataset.isin(training_cls)]["accuracy"].ravel()
    if output_normalization:
        self.mean_output = np.mean(y_train).astype(np.float32)
        self.std_output = np.std(y_train).astype(np.float32)
        try:
            assert self.std_output !=0.
        except Exception:
            self.std_output = 1.
    else:
        self.mean_output = 0.
        self.std_output = 1.

    self.x = {"train":torch.tensor(X_train.astype(np.float32))}
    self.y = {"train":torch.tensor(y_train.astype(np.float32))}

def get_tr_loader(batch_size, data_path, loo, mode,use_meta=True, cv=None, split_type="cv", sparsity = 0):
    
  if split_type=="cv":
      dataset = TrainDatabaseCV(data_path, loo,mode=mode)
  else:
      dataset = TrainDatabaseCVPlusLoo(data_path, cv=cv,loo=loo,mode=mode,sparsity = sparsity,use_meta=use_meta)
  loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
  unshuffled_loader = DataLoader(dataset=copy.deepcopy(dataset),batch_size=525,shuffle=False)
  unshuffled_loader.dataset.mode="regression"
  return loader,unshuffled_loader

def get_ts_loader(batch_size, data_path, loo, mu_in,std_in,mu_out,std_out, split_type="cv",use_meta=True):
  dataset = TestDatabase(data_path, loo,mu_in,std_in,mu_out,std_out, split_type=split_type,use_meta=use_meta)
  loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)
  return loader

def get_ts_loader_online(batch_size, data, loo, mu_in, std_in, split_type="cv",use_meta=True):
  dataset = TestDatabaseOnline(data, loo, mu_in, std_in, split_type=split_type,use_meta=use_meta)
  loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
  return loader 

# mtrloader,mtrloader_unshuffled =  get_tr_loader(64, "data", 1, 
                                        # mode="bpr",split_type="cv")


