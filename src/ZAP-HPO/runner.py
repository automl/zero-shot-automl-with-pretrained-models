#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import ndcg_score
import os
import json
import pickle
import numpy as np
import time
from tqdm import tqdm
from utils import Log, get_log, save_model, config_from_yaml, config_to_yaml, construct_model_path
from loader import get_tr_loader, get_ts_loader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class surrogate(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu", dropout=0.0):
        
        super(surrogate, self).__init__()
        
        assert(nonlinearity == "relu")
        self.nonlinearity = nn.ReLU()
        
        self.fc = nn.ModuleList([nn.Linear(in_features = d_in, out_features = output_sizes[0])])
        for d_out in output_sizes[1:]:
            self.fc.append(nn.Linear(in_features = self.fc[-1].out_features, out_features = d_out))

        self.dropout = nn.Dropout(dropout)

        self.out_features = output_sizes[-1]
    
    def forward(self, x):
        
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        return x

def WMSE(input, target, weights):
    out = (input-target)**2
    out = out * weights
    return out.mean()

class ModelRunner:
    def __init__(self, args):

        torch.manual_seed(args.seed)

        self.args = args
        self.seed = args.seed
        self.config_seed = args.config_seed
        self.save_path = args.save_path
        self.data_path = args.data_path
        self.config_path = args.config_path
        self.save_epoch = args.save_epoch
        self.max_epoch = args.max_epoch
        self.mode = args.mode
        self.weighted = args.weighted
        self.weigh_fn = args.weigh_fn
        self.split_type = args.split_type
        self.loo =  args.loo
        self.cv = args.cv
        self.sparsity = args.sparsity
        self.use_meta = args.use_meta
        self.num_aug = args.num_aug
        self.num_pipelines = args.num_pipelines
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_corr_dict = {'rank@1': np.inf, 'epoch': -1, "ndcg@5":-1, "ndcg@10":-1, "ndcg@20":-1}
        
        # check for args.config_path
        if self.config_path is None:
            cs = self.get_configspace(self.config_seed)
            self.config = cs.sample_configuration()
        else:
            self.config = config_from_yaml(self.config_path)

        self.model = surrogate(d_in = 39 if self.use_meta else 35, 
                               output_sizes = self.config["num_hidden_layers"]*[self.config["num_hidden_units"]]+[1], 
                               dropout = self.config["dropout_rate"])
        self.model.to(self.device)
        
        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum = self.config['sgd_momentum'], weight_decay = self.config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.max_epoch, eta_min = self.config["min_lr"])

        self.mtrloader, self.mtrloader_unshuffled =  get_tr_loader(seed = self.seed, 
                                                                   data_path = self.data_path, 
                                                                   mode = self.mode,
                                                                   split_type = self.split_type,
                                                                   cv = self.cv,
                                                                   loo = self.loo, 
                                                                   sparsity = self.sparsity,
                                                                   use_meta = self.use_meta,
                                                                   num_aug = self.num_aug, 
                                                                   num_pipelines = self.num_pipelines,
                                                                   batch_size = self.config['batch_size'])

        self.model_path = construct_model_path(self.save_path, self.config_seed, self.split_type, self.loo, self.cv, 
                                               self.mode, self.weighted, self.weigh_fn, self.sparsity, self.use_meta)
        
        os.makedirs(self.model_path, exist_ok=True)
        self.mtrlog = Log(self.args, open(os.path.join(self.model_path, 'meta_train_predictor.log'), 'w'))
        self.mtrlog.print_args(self.config)  

        with open(os.path.join(self.model_path, "input_scaler.pt"), 'wb') as f:
            pickle.dump(self.mtrloader.dataset.input_scaler, f) 

        with open(os.path.join(self.model_path, "output_scaler.pt"), 'wb') as f:
            pickle.dump(self.mtrloader.dataset.output_scaler, f) 

        config_to_yaml(os.path.join(self.model_path, "model_config.yaml"), self.config)
    
    def train(self):
        history = {"trndcg": [], "vandcg": []}
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            if self.mode=="regression":
                loss = self.train_epoch()  
            elif self.mode=="bpr":
                loss = self.train_bpr_epoch()
            elif self.mode=="tml":
                loss = self.train_tml_epoch()

            self.scheduler.step()

            self.mtrlog.print_pred_log(loss, 0, 'train', epoch=epoch)

            vacorr, vaccc, vandcg = self.validation("valid")
            trcorr, tracc, trndcg = self.validation("train")
 
            if self.max_corr_dict['rank@1'] >= vacorr:
                patience = 0

            if self.max_corr_dict['rank@1'] > vacorr:
                self.max_corr_dict['rank@1'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict.update(vandcg)
                save_model(epoch, self.model, self.model_path, max_corr=True)


            self.mtrlog.print_pred_log(0, vacorr, 'valid', ndcg=vandcg, max_corr_dict=self.max_corr_dict)
            
            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)
            
            vandcg.update({"acc":vaccc,
                           "rank":vacorr})                          
            trndcg.update({"acc":tracc,
                           "rank":trcorr,
                           "loss":loss})              
 
            history["trndcg"].append(trndcg)
            history["vandcg"].append(vandcg)  

            history_path = os.path.join(self.model_path, "history.json")      
            with open(history_path, 'w') as f:
                json.dump(history, f)  

        self.mtrlog.save_time_log()

        return history
        
    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)

        self.mtrloader.dataset.training="train"
        dlen = 0
        trloss = 0
        pbar = self.mtrloader
        for x, acc, y_ in pbar:
            x = x.to(self.device)
            y = acc.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            
            if not self.weighted:
              loss = nn.MSELoss()(y_pred, y.unsqueeze(-1))
            else:
              loss = WMSE(y_pred,y.unsqueeze(-1),weights=torch.exp(-(acc-y_).pow(2)).unsqueeze(-1))

            loss.backward()
            self.optimizer.step()
    
            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            trloss += float(loss)
            dlen+=1
    
        return trloss/dlen

    def calculate_bpr_loss(self, acc, acc_s, acc_l, r, r_s, r_l,logits):
        if self.weighted:
            if self.weigh_fn == "v0":
                weights = torch.cat([torch.exp(-(acc - acc_s).pow(2)),
                                     torch.exp(-(acc_s - acc).pow(2)),
                                     torch.exp(-(acc_l - acc_s).pow(2))], 0)
            elif self.weigh_fn == "v1":
                weights = torch.cat([(acc - acc_s).pow(2),
                                     (acc_l - acc).pow(2),
                                     (acc_l - acc_s).pow(2)], 0)
            elif self.weigh_fn == "v0-rank":
                weights = torch.cat([torch.exp(-((r - r_s)).pow(2)),
                                     torch.exp(-((r_l - r)).pow(2)),
                                     torch.exp(-((r_l - r_s)).pow(2))], 0)
            elif self.weigh_fn == "v1-rank":
                weights = torch.cat([((r - r_s)).pow(2),
                                     ((r_l - r)).pow(2),
                                     ((r_l - r_s)).pow(2)], 0)

            return nn.BCELoss(weight=weights.unsqueeze(-1))(logits, torch.ones_like(logits))
        else:
            return nn.BCELoss()(logits, torch.ones_like(logits).to(self.device))

    def train_bpr_epoch(self):
        self.model.train()
        self.model.to(self.device)

        dlen = 0
        trloss = 0
        pbar = self.mtrloader

        for (x, s, l), (acc, acc_s, acc_l), (r, r_s, r_l) in pbar:
            
            x = x.to(self.device)
            s = s.to(self.device)
            l = l.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model.forward(x)
            y_pred_s = self.model.forward(s)
            y_pred_l = self.model.forward(l) 

            output_gr_smaller = nn.Sigmoid()(y_pred - y_pred_s)
            larger_gr_output  = nn.Sigmoid()(y_pred_l - y_pred)
            larger_gr_smaller  = nn.Sigmoid()(y_pred_l - y_pred_s)

            logits = torch.cat([output_gr_smaller,larger_gr_output,larger_gr_smaller], 0) # concatenates end to end

            loss = self.calculate_bpr_loss(acc, acc_s, acc_l, r, r_s, r_l, logits)

            loss.backward()
            self.optimizer.step()

            trloss += float(loss)
            dlen+=1

        return trloss/dlen

    def train_tml_epoch(self, margin = 1.0):
        self.model.train()
        self.model.to(self.device)

        dlen = 0
        trloss = 0
        pbar = self.mtrloader

        for (x, s, l), (acc,acc_s,acc_l), (r,r_s,r_l) in pbar:

            x = x.to(self.device)
            s = s.to(self.device)
            l = l.to(self.device)
            
            self.optimizer.zero_grad()

            # perf predictions for target, inferior, superior configurations
            # range [0, 1]
            y_pred = self.model.forward(x) 
            y_pred_s = self.model.forward(s)
            y_pred_l = self.model.forward(l) 

            loss = nn.TripletMarginLoss(margin = margin)(y_pred, y_pred_l, y_pred_s)
            
            loss.backward()
            self.optimizer.step()

            trloss += float(loss)
            dlen+=1

        return trloss/dlen    

    def validation(self, training):
        self.model.eval()
        self.model.to(self.device)

        self.mtrloader_unshuffled.dataset.training=training
        pbar = self.mtrloader_unshuffled
        scores_5 = []
        scores_10 = []
        scores_20 = []
        ranks = []
        values = []
        predicted_y =[] 
        actual_y = []
        with torch.no_grad():
          for i,(x,acc,y_) in enumerate(pbar):
            x = x.to(self.device)
            y = acc.to(self.device).tolist()
            y_pred = self.model.forward(x).squeeze().tolist()
            
            actual_y += y
            predicted_y += y_pred
            if training!="train": # batch_isez is fixed
                scores_5.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=5))
                scores_10.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=10))
                scores_20.append(ndcg_score(y_true=np.array(y).reshape(1,-1),y_score=np.maximum(1e-7,np.array(y_pred)).reshape(1,-1),k=20))
                ranks.append(self.mtrloader_unshuffled.dataset.ranks[training][i][np.argmax(y_pred)])
                values.append(self.mtrloader_unshuffled.dataset.values[training][i][np.argmax(y_pred)])            
        if training=="train":
            start = 0
            for i,ds in enumerate(np.unique(self.mtrloader_unshuffled.dataset.ds_ref)):
                end = start + len(np.where(self.mtrloader_unshuffled.dataset.ds_ref==ds)[0])
                y_true = np.array(actual_y[start:end]).reshape(1,-1)
                y_score=np.maximum(1e-7,np.array(predicted_y[start:end])).reshape(1,-1)
                y_pred = predicted_y[start:end]
                scores_5.append(ndcg_score(y_true=y_true,y_score=y_score,k=5))
                scores_10.append(ndcg_score(y_true=y_true,y_score=y_score,k=10))
                scores_20.append(ndcg_score(y_true=y_true,y_score=y_score,k=20))
                ranks.append(self.mtrloader_unshuffled.dataset.ranks[training][i][np.argmax(y_pred)])
                values.append(self.mtrloader_unshuffled.dataset.values[training][i][np.argmax(y_pred)])
                start = end

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
    parser.add_argument('--seed', type=int, default=0, 
                        help = "Seed for sampling from the meta-dataset")
    parser.add_argument('--config_seed', type=int, default=0, 
                        help="Seed for sampling a surrogate config")
    parser.add_argument('--save_path', type=str, default='../ckpts', 
                        help="The path of the model/log save directory")
    parser.add_argument('--data_path', type=str, default='../../data', 
                        help="The path of the metadata directory")
    parser.add_argument('--config_path',type=str, 
                        help='Path to config stored in yaml file. No value implies the CS will be sampled.')
    parser.add_argument('--save_epoch', type=int, default=20, 
                        help="How many epochs to wait each time to save model states") 
    parser.add_argument('--max_epoch', type=int, default=400, 
                        help="Number of epochs to train")
    parser.add_argument('--mode', type=str, default='bpr', choices=["regression", "bpr", "tml"],
                        help="Training objective. Choices: regression|bpr|tml")
    parser.add_argument('--weighted', type=str, default="False", choices=["True","False"],
                        help="Whether to use the weighted objective. Only used when the training objective is regression or BPR.")
    parser.add_argument('--weigh_fn', type=str, default="v0", choices=["v0","v1", "v0-rank", "v1-rank"],
                        help="BPR objective weighing fn. Only used when '--weighted' is 'True'.")
    parser.add_argument('--split_type', type=str, default="cv", 
                        help="When loo, this omits the designated core-dataset augmentations from the training procedure. Choices: cv|loo")
    parser.add_argument('--loo', type=int, default=0, 
                        help="Index of the core dataset [0,34] that should be removed")
    parser.add_argument('--cv', type=int, default=1, 
                        help="Index of CV [1,5]. Remark that this is the inner split. If LOO, respective core dataset augmentations will be removed from its CV fold.")
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help="Proportion [0.0,1.0) of the missing values in the meta-dataset.")
    parser.add_argument('--use_meta', type=str, default="True", choices=["True","False"],
                        help="Whether to use the dataset meta-features.")    
    parser.add_argument('--num_aug', type=int, default=15, 
                        help="The number of ICGen augmentations per dataset.")
    parser.add_argument('--num_pipelines', type=int, default=525, 
                        help="The number of deep learning pipelines.")

    args = parser.parse_args()
    args.weighted = eval(args.weighted)
    args.use_meta = eval(args.use_meta)

    runner = ModelRunner(args)
    history = runner.train()

    history_path = os.path.join(runner.model_path, "history.json")      
    with open(history_path, 'w') as f:
        json.dump(history, f)

