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
from loader import get_tr_loader,get_ts_loader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from runner import batch_mlp
from utils import load_model, config_from_yaml

class ModelTester:
    def __init__(self,args):

        self.args = args
        self.data_path = args.data_path
        self.loo =  args.loo
        self.cv = args.cv
        self.num_aug = args.num_aug
        self.num_pipelines = args.num_pipelines
        self.mode = args.mode
        self.seed = args.seed
        self.sparsity = args.sparsity
        self.use_meta = args.use_meta
        self.save_path = args.save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check for args.config_path
        if args.config_path is None:
            cs = self.get_configspace(self.seed)
            config = cs.sample_configuration()
        else:
            config = config_from_yaml(args.config_path)

        self.model = batch_mlp(d_in=39 if self.use_meta else 35,output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1],
                               dropout=config["dropout_rate"])
        self.model.to(self.device)
        
        extra = f"-{self.sparsity}" if self.sparsity > 0 else ""
        extra += "-no-meta" if not self.use_meta else ""
        if self.mode == "bpr":
            extra += f"-function-{self.fn}" if self.weighted else ""

        if self.split_type!="loo":
            self.model_path = os.path.join(self.save_path, f"{'weighted-' if self.weighted else ''}{self.mode}{extra}", str(self.seed), str(self.cv))
        else:
            self.model_path = os.path.join(self.save_path+"-cvplusloo", f"{'weighted-' if self.weighted else ''}{self.mode}{extra}", str(self.seed), str(self.loo),str(self.cv))
        
            
        if args.load_model:
            load_model(self.model, device=self.device, model_path = self.model_path)

        with open(os.path.join(self.model_path, "input_scaler.pt"), 'rb') as f:
            self.input_scaler = pickle.load(f) 

        with open(os.path.join(self.model_path, "output_scaler.pt"), 'rb') as f:
            self.output_scaler = pickle.load(f)

        self.mtrloader_test =  get_ts_loader(self.data_path, self.loo,
                                             input_scaler = self.input_scaler,
                                             output_scaler = self.output_scaler,
                                             use_meta=self.use_meta,
                                             num_aug = self.num_aug, num_pipelines = self.num_pipelines)
        

    def test(self):
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../ckpts', help='the path of save directory')
    parser.add_argument('--data-path', type=str, default='../../data', help='the path of save directory')
    parser.add_argument('--load-model', type=str, default="True", choices=["True","False"])
    parser.add_argument('--mode', type=str, default='bpr', help='training objective',choices=["regression","bpr"])
    parser.add_argument('--loo', type=int, default=0, help='Index of dataset [0,34] that should be removed')
    parser.add_argument('--cv', type=int, default=1, help='Index of CV [1,5]')
    parser.add_argument('--split_type', type=str, default="loo", help='cv|loo')
    parser.add_argument('--sparsity', type=float, default=0.)
    parser.add_argument('--use-meta', type=str, default="True", choices=["True","False"])
    parser.add_argument('--config_path',type=str, help= 'Path to config stored in yaml file. No value implies the CS will be sampled')
    parser.add_argument('--num_aug', type=int, default=15, help='The number of ICGen augmentations per dataset')
    parser.add_argument('--num_pipelines', type=int, default=525, help='The number of deep learning pipelines')
    
    args = parser.parse_args()
    args.use_meta = eval(args.use_meta)
    args.load_model = eval(args.load_model)
    
    runner = ModelTester(args)
    scores  = runner.test()
    names = []
    for i in runner.mtrloader_test.dataset.testing_cls:
        names += [i]*args.num_pipelines
    data = pd.DataFrame(names, columns=["dataset"])
    data["scores"] = scores
