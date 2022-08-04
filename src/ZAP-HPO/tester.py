#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from loader import get_tr_loader,get_ts_loader
from runner import surrogate
from utils import load_model, config_from_yaml

class ModelTester:
    def __init__(self,args):

        self.args = args
        self.config_seed = args.config_seed
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.config_path = args.config_path
        self.load_model = args.load_model
        self.loo =  args.loo
        self.use_meta = args.use_meta
        self.num_aug = args.num_aug
        self.num_pipelines = args.num_pipelines
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_path is None:
            cs = self.get_configspace(self.config_seed)
            config = cs.sample_configuration()
        else:
            config = config_from_yaml(self.config_path)

        self.model = surrogate(d_in=39 if self.use_meta else 35,
                               output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1],
                               dropout=config["dropout_rate"])
        self.model.to(self.device)
        
        if self.load_model:
            load_model(self.model, device=self.device, model_path = self.model_path)

            with open(os.path.join(self.model_path, "input_scaler.pkl"), 'rb') as f:
                self.input_scaler = pickle.load(f) 

            with open(os.path.join(self.model_path, "output_scaler.pkl"), 'rb') as f:
                self.output_scaler = pickle.load(f)
        else:
            self.input_scaler = None
            self.output_scaler = None

        self.mtrloader_test =  get_ts_loader(self.data_path, self.loo,
                                             input_scaler = self.input_scaler,
                                             output_scaler = self.output_scaler,
                                             use_meta=self.use_meta,
                                             num_aug = self.num_aug, num_pipelines = self.num_pipelines)
        

    def predict(self):
        self.model.to(self.device)
        self.model.eval()
        pbar = tqdm(self.mtrloader_test)
        predicted_y = []
        with torch.no_grad():
          for i,(x,acc) in enumerate(pbar):
            x = x.to(self.device)
            y = acc.to(self.device)
            y_pred = self.model.forward(x)
            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            predicted_y+=y_pred
        return predicted_y

    def test(self):
        self.model.eval()
        self.model.to(self.device)

        pbar = self.mtrloader_test
        scores_5 = []
        scores_10 = []
        scores_20 = []
        ranks = []
        values = []
        with torch.no_grad():
          for i,(x,acc) in enumerate(pbar):
            x = x.to(self.device)
            y = acc.to(self.device).tolist()
            y_pred = self.model.forward(x)
            y_pred = y_pred.squeeze().tolist()
            scores_5.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=5))
            scores_10.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=10))
            scores_20.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=20))
            ranks.append(self.mtrloader_test.dataset.ranks[i][np.argmax(y_pred)])
            values.append(self.mtrloader_test.dataset.values[i][np.argmax(y_pred)])
         
        return np.mean(ranks),np.mean(values), {"NDCG@5":np.mean(scores_5),"NDCG@10":np.mean(scores_10),"NDCG@20":np.mean(scores_20)}



    @staticmethod
    def get_configspace(seed):
        
        cs = CS.ConfigurationSpace(seed=seed)
        
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-2, log=True)
        min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-8, upper=1e-6, log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD', 'AdamW'])
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-2, log=True)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=8, upper=128, default_value=64)
        cs.add_hyperparameters([lr, min_lr, optimizer, weight_decay, batch_size])
        
        num_hidden_layers =  CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=2, upper=10)
        num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=32, upper=512)
        cs.add_hyperparameters([num_hidden_layers, num_hidden_units])
        
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, log=False)
        cs.add_hyperparameters([dropout_rate])
        
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, log=False)
        cs.add_hyperparameters([sgd_momentum])
        momentum_cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_conditions([momentum_cond])
        
        return cs
    
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_seed', type=int, default=0, 
                        help="Seed for sampling a surrogate config.")
    parser.add_argument('--model_path', type=str, 
                        help="The full path of the model parent directory.")
    parser.add_argument('--data_path', type=str, default='../../data/meta_dataset', 
                        help="The path of the metadata directory")
    parser.add_argument('--config_path',type=str, default = "default_config.yaml", 
                        help='Path to config stored in yaml file. No value implies the CS will be sampled.')
    parser.add_argument('--load_model', type=str, default="True", choices=["True","False"],
                        help="Used for debugging purposes without having a model state.")
    parser.add_argument('--loo', type=int, default=0, 
                        help="Index of the core dataset [0,34] that was removed and should be tested here.")
    parser.add_argument('--use_meta', type=str, default="True", choices=["True","False"],
                        help="Whether to use the dataset meta-features.")    
    parser.add_argument('--num_aug', type=int, default=15, 
                        help="The number of ICGen augmentations per dataset.")
    parser.add_argument('--num_pipelines', type=int, default=525, 
                        help="The number of deep learning pipelines.")
    
    args = parser.parse_args()
    args.use_meta = eval(args.use_meta)
    args.load_model = eval(args.load_model)
    
    runner = ModelTester(args)
    predictions = runner.predict()

    names = []
    for i in runner.mtrloader_test.dataset.test_datasets:
        names += [i]*args.num_pipelines
    data = pd.DataFrame(names, columns=["dataset"])
    data["predictions"] = predictions

    tecorr, teacc, tendcg = runner.test()
    print(f"Mean test rank: {tecorr}")
    for recall, ndcg_score in tendcg.items():
        print(f"{recall}: {ndcg_score}")

    # --model_path ../../data/models/ZAP-HPO/bpr-loo/0/1 --loo 0
    # --model_path ../ckpts-cvplusloo/bpr/default_config/0/1 --loo 0