#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from loader import get_ts_loader, get_pred_loader
from runner import surrogate
from utils import load_model, config_from_yaml

class ModelTester:
    def __init__(self, model_path, data_path, loo, use_meta, num_aug, num_pipelines):

        self.model_path = model_path
        self.data_path = data_path
        self.loo = loo
        self.use_meta = use_meta
        self.num_aug = num_aug
        self.num_pipelines = num_pipelines
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config_from_yaml(os.path.join(model_path, "model_config.yaml"))

        neurons_per_layer = [self.config["num_hidden_units"] for _ in range(self.config["num_hidden_layers"])] # hidden layers
        neurons_per_layer.append(1) # output layer
        self.model = surrogate(d_in=39 if self.use_meta else 35,
                               output_sizes=neurons_per_layer,
                               dropout=self.config["dropout_rate"])
        self.model.to(self.device)
        
        load_model(self.model, device=self.device, model_path = self.model_path)

        with open(os.path.join(self.model_path, "input_scaler.pkl"), 'rb') as f:
            self.input_scaler = pickle.load(f) 
        with open(os.path.join(self.model_path, "output_scaler.pkl"), 'rb') as f:
            self.output_scaler = pickle.load(f)

        self.mtrloader_test =  get_ts_loader(self.data_path, 
                                             self.loo,
                                             self.input_scaler,
                                             self.output_scaler,
                                             self.use_meta,
                                             self.num_aug, 
                                             self.num_pipelines)
        

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        predicted_y = []
        with torch.no_grad():
          for x,_ in self.mtrloader_test:
            x = x.to(self.device)
            y_pred = self.model.forward(x)
            y_pred = y_pred.squeeze().tolist()
            predicted_y+=y_pred
        return predicted_y

    def test(self):
        self.model.eval()
        self.model.to(self.device)

        scores_5 = []
        scores_10 = []
        scores_20 = []
        ranks = []
        values = []
        with torch.no_grad():
          for i,(x,acc) in enumerate(self.mtrloader_test):
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

class ModelPredictor:
    def __init__(self, model_path, data, use_meta):

        self.model_path = model_path
        self.data = data
        self.use_meta = use_meta
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config_from_yaml(os.path.join(self.model_path, "model_config.yaml"))

        neurons_per_layer = [self.config["num_hidden_units"] for _ in range(self.config["num_hidden_layers"])] # hidden layers
        neurons_per_layer.append(1) # output layer
        self.model = surrogate(d_in=39 if self.use_meta else 35,
                               output_sizes=neurons_per_layer,
                               dropout=self.config["dropout_rate"])
        self.model.to(self.device)
        
        load_model(self.model, device=self.device, model_path = self.model_path)

        with open(os.path.join(self.model_path, "input_scaler.pkl"), 'rb') as f:
            self.input_scaler = pickle.load(f) 

        self.mtrloader_test =  get_pred_loader(self.data, self.input_scaler, self.use_meta)

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        predicted_y = []
        with torch.no_grad():
          for x in self.mtrloader_test:
            x = x.to(self.device)
            y_pred = self.model.forward(x)
            y_pred = y_pred.squeeze().tolist()
            predicted_y+=y_pred
        return predicted_y


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        help="The full path of the model parent directory.")
    parser.add_argument('--data_path', type=str, default='../../data/meta_dataset', 
                        help="The path of the metadata directory")
    parser.add_argument('--loo', type=str, default="cifar10", 
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
    
    runner = ModelTester(args.model_path, args.data_path, args.loo, args.use_meta, args.num_aug, args.num_pipelines)
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
