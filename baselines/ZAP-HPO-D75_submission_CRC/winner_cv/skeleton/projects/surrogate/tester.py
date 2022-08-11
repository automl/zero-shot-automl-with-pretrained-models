#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:21:55 2022

@author: hsjomaa
"""
try:
	import torch
	import torch.utils.data
except:
	raise ImportError("For this example you need to install pytorch.")

import os
import pandas as pd
from tqdm import tqdm
from loader import get_tr_loader,get_ts_loader, get_ts_loader_online
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from runner import batch_mlp
from utils import load_model
class ModelTester:
    def __init__(self, args):

        self.args = args
        self.data_path = args.data_path
        self.loo =  args.loo
        self.cv = args.cv
        self.mode = args.mode
        self.seed = args.seed
        self.sparsity = args.sparsity
        self.use_meta = args.use_meta
        self.save_path = args.save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cs = self.get_configspace(self.seed)
        config = cs.sample_configuration()
        self.model = batch_mlp(d_in=39 if self.use_meta else 35,output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1],
                               dropout=config["dropout_rate"])
        self.model.to(self.device)
        extra = f"-{self.sparsity}" if self.sparsity > 0 else ""
        extra += "-no-meta" if not self.use_meta else ""
        self.model_path = os.path.join(self.save_path, f"{self.mode}{extra}", str(self.seed), str(self.loo),str(self.cv))
            
        if args.load_model:
            load_model(self.model,device=self.device, model_path = self.model_path)
        self.mtrloader,self.mtrloader_unshuffled =  get_tr_loader(64, self.data_path, loo=self.loo, cv=self.cv,
                                        mode=self.mode,split_type=args.split_type,sparsity =self.sparsity,
                                        use_meta=self.use_meta)
        self.mtrloader_test =  get_ts_loader(525, self.data_path, self.loo,
                                              mu_in=self.mtrloader.dataset.mean_input,
                                              std_in=self.mtrloader.dataset.std_input,
                                              mu_out=self.mtrloader.dataset.mean_output,
                                              std_out=self.mtrloader.dataset.std_output,split_type=args.split_type,
                                              use_meta=self.use_meta)
        

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        pbar = tqdm(self.mtrloader_test)
        predicted_y = []
        with torch.no_grad():
          for i,(x,acc) in enumerate(pbar):
            y_pred = self.model.forward(x)
            y = acc.to(self.device)
            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            predicted_y+=y_pred
        return predicted_y


    @staticmethod
    def get_configspace(seed):
        cs = CS.ConfigurationSpace(seed=seed)
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
		# For demonstration purposes, we add different optimizers as categorical hyperparameters.
		# To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
		# SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam'])
        cs.add_hyperparameters([lr, optimizer])
        num_hidden_layers =  CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=5, upper=10, default_value=5)
        num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=32, upper=512, default_value=64)
        cs.add_hyperparameters([num_hidden_layers, num_hidden_units])
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
        cs.add_hyperparameters([dropout_rate])
        return cs    
    

class ModelTesterOnline:
    def __init__(self, data_path, test_data, loo, cv, mode, seed, sparsity, use_meta, save_path, split_type):

        self.data_path = data_path
        self.test_data = test_data
        self.loo =  loo
        self.cv = cv
        self.mode = mode
        self.seed = seed
        self.sparsity = sparsity
        self.use_meta = use_meta
        self.save_path = save_path
        self.device = torch.device("cpu")
        cs = self.get_configspace(self.seed)
        config = cs.sample_configuration()
        self.model = batch_mlp(d_in=39 if self.use_meta else 35,output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1],
                               dropout=config["dropout_rate"])
        self.model.to(self.device)
        extra = f"-{self.sparsity}" if self.sparsity > 0 else ""
        extra += "-no-meta" if not self.use_meta else ""
        self.model_path = os.path.join(self.save_path,str(self.cv))

        self.mtrloader, _ =  get_tr_loader(64, self.data_path, loo=self.loo, cv=self.cv,
                                           mode=self.mode,split_type=split_type,sparsity =self.sparsity,
                                           use_meta=self.use_meta)
        self.mtrloader_test =  get_ts_loader_online(525, self.test_data, self.loo,
                                             mu_in=self.mtrloader.dataset.mean_input,
                                             std_in=self.mtrloader.dataset.std_input,
                                             split_type=split_type,
                                             use_meta=self.use_meta)

        load_model(self.model,device=self.device, model_path = self.model_path)

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        pbar = tqdm(self.mtrloader_test)
        predicted_y = []
        with torch.no_grad():
          for i, x in enumerate(pbar):
            y_pred = self.model.forward(x)
            y_pred = y_pred.squeeze().tolist()
            predicted_y+=y_pred
        return predicted_y


    @staticmethod
    def get_configspace(seed):
        cs = CS.ConfigurationSpace(seed=seed)
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam'])
        cs.add_hyperparameters([lr, optimizer])
        num_hidden_layers =  CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=5, upper=10, default_value=5)
        num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=32, upper=512, default_value=64)
        cs.add_hyperparameters([num_hidden_layers, num_hidden_units])
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
        cs.add_hyperparameters([dropout_rate])
        return cs    

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--data-path', type=str, default='../data', help='the path of save directory')
    parser.add_argument('--save-path', type=str, default='.data/ZAP-HPO/ckpts/ZAP-HPO-D75_submission_CRC', help='the path of save directory')
    parser.add_argument('--load-model', type=str, default="True", choices=["True","False"])
    parser.add_argument('--mode', type=str, default='bpr', help='training objective',choices=["regression","bpr"])
    parser.add_argument('--loo', type=int, default=0, help='Index of dataset [0,34] that should be removed')
    parser.add_argument('--cv', type=int, default=1, help='Index of CV [1,5]')
    parser.add_argument('--split_type', type=str, default="loo", help='cv|loo')
    parser.add_argument('--sparsity', type=float, default=0.)
    parser.add_argument('--use-meta', type=str, default="True", choices=["True","False"])    
    args = parser.parse_args()
    args.use_meta = eval(args.use_meta)
    args.load_model = eval(args.load_model)
    runner = ModelTester(args)
    scores  = runner.test()
    names = []
    for i in runner.mtrloader_test.dataset.testing_cls:
        names += [i]*525
    data = pd.DataFrame(names,columns=["dataset"])
    data["scores"] = scores
