#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import ndcg_score
import os
import pickle
import numpy as np
import time
from tqdm import tqdm
from utils import Log, save_model, config_from_yaml, config_to_yaml, construct_model_path
from loader import get_tr_loader, get_ts_loader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class surrogate(nn.Module):
    '''
    The surrogate MLP model class
    '''
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
    '''
    The weighted loss fn for the regression
    '''
    out = (input-target)**2
    out = out * weights
    return out.mean()

def weighted_bpr_loss(logits, accuracies, ranks, weigh_fn = "v1", device = torch.device("cpu")):
    '''
    The weighted loss fn for the binary pairwise ranking
    '''
    acc, acc_s, acc_l = accuracies
    r, r_s, r_l = ranks

    if weigh_fn == "v0":
        weights = torch.cat([torch.exp(-(acc - acc_s).pow(2)),
                             torch.exp(-(acc_s - acc).pow(2)),
                             torch.exp(-(acc_l - acc_s).pow(2))], 0)
    elif weigh_fn == "v1":
        weights = torch.cat([(acc - acc_s).pow(2),
                             (acc_l - acc).pow(2),
                             (acc_l - acc_s).pow(2)], 0)
    elif weigh_fn == "v0-rank":
        weights = torch.cat([torch.exp(-((r - r_s)).pow(2)),
                             torch.exp(-((r_s - r)).pow(2)),
                             torch.exp(-((r_l - r_s)).pow(2))], 0)
    elif weigh_fn == "v1-rank":
        weights = torch.cat([((r - r_s)).pow(2),
                             ((r_l - r)).pow(2),
                             ((r_l - r_s)).pow(2)], 0)
    weights = weights.unsqueeze(-1).to(device)

    return nn.BCELoss(weight=weights)(logits, torch.ones_like(logits).to(device))


class ModelRunner:
    '''
    The surrogate training/validation class
    '''
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
        
        # Check for args.config_path
        if self.config_path is None:
            cs = self.get_configspace(self.config_seed)
            self.config = cs.sample_configuration()
            self.config_identifier = "seed-"+str(self.config_seed)
        else:
            self.config = config_from_yaml(self.config_path)
            self.config_identifier = self.config_path.split("/")[-1].split(".yaml")[0]

        neurons_per_layer = [self.config["num_hidden_units"] for _ in range(self.config["num_hidden_layers"])] # hidden layers
        neurons_per_layer.append(1) # output layer
        self.model = surrogate(d_in = 39 if self.use_meta else 35, 
                               output_sizes = neurons_per_layer,
                               dropout = self.config["dropout_rate"])
        self.model.to(self.device)
        
        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum = self.config['sgd_momentum'], weight_decay = self.config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.max_epoch, eta_min = self.config["min_lr"])

        # Get the meta-training dataset. Also contains the validation set
        self.mtrloader, self.mtrloader_unshuffled =  get_tr_loader(self.seed, 
                                                                   self.data_path, 
                                                                   self.mode,
                                                                   self.split_type,
                                                                   self.cv,
                                                                   self.loo, 
                                                                   self.sparsity,
                                                                   self.use_meta,
                                                                   self.num_aug, 
                                                                   self.num_pipelines,
                                                                   self.config['batch_size'])


        self.model_path = construct_model_path(self.save_path, self.config_identifier, self.split_type, self.loo, self.cv, 
                                               self.mode, self.weighted, self.weigh_fn, self.sparsity, self.use_meta)
        os.makedirs(self.model_path, exist_ok=True)
        self.history_path = os.path.join(self.model_path, "history.pkl")

        self.mtrlog = Log(self.args, open(os.path.join(self.model_path, 'meta_train_predictor.log'), 'w'))
        self.mtrlog.print_args(self.config)  

        # Save the meta-dataset scalers for later use on test
        with open(os.path.join(self.model_path, "input_scaler.pkl"), 'wb') as f:
            pickle.dump(self.mtrloader.dataset.input_scaler, f) 
        with open(os.path.join(self.model_path, "output_scaler.pkl"), 'wb') as f:
            pickle.dump(self.mtrloader.dataset.output_scaler, f) 

        # Save the surrogate model's config for later initialization on test
        config_to_yaml(os.path.join(self.model_path, "model_config.yaml"), self.config)
    
    def train(self):
        '''
        The main training wrapper.
        '''
        history = {"trndcg": [], "vandcg": []}
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            if self.mode=="regression":
                loss = self.train_regression_epoch()  
            elif self.mode=="bpr":
                loss = self.train_bpr_epoch()
            elif self.mode=="tml":
                loss = self.train_tml_epoch()

            self.scheduler.step()

            vacorr, vaccc, vandcg = self.validation("valid")
            trcorr, tracc, trndcg = self.validation("train")

            self.mtrlog.print_pred_log(loss, 0, 'train', epoch=epoch)
            self.mtrlog.print_pred_log(0, vacorr, 'valid', ndcg=vandcg, max_corr_dict=self.max_corr_dict)

            vandcg.update({"acc":vaccc,
                           "rank":vacorr})                          
            trndcg.update({"acc":tracc,
                           "rank":trcorr,
                           "loss":loss})              
            
            # Update the training history
            history["trndcg"].append(trndcg)
            history["vandcg"].append(vandcg)  
            with open(self.history_path, 'wb') as f:
                pickle.dump(history, f) 

            # Save the best surrogate model
            if self.max_corr_dict['rank@1'] > vacorr:
                self.max_corr_dict['rank@1'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict.update(vandcg)
                save_model(epoch, self.model, self.model_path, max_corr=True)

            # Save a surrogate model checkpoint in every few epochs
            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)
            
        self.mtrlog.save_time_log()

        
    def train_regression_epoch(self):
        self.model.train()
        self.model.to(self.device)
        
        dlen = 0
        trloss = 0
        for x, acc, y_ in self.mtrloader:
            x = x.to(self.device)
            y = acc.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            
            if not self.weighted:
                loss = nn.MSELoss()(y_pred, y.unsqueeze(-1))
            else:
                weights = torch.exp(-(acc-y_).pow(2)).unsqueeze(-1).to(self.device)
                loss = WMSE(y_pred, y.unsqueeze(-1), weights)

            loss.backward()
            self.optimizer.step()
    
            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            trloss += float(loss)
            dlen+=1
    
        return trloss/dlen


    def train_bpr_epoch(self):
        self.model.train()
        self.model.to(self.device)

        dlen = 0
        trloss = 0
        for (x, s, l), accuracies, ranks in self.mtrloader:
            
            x = x.to(self.device)
            s = s.to(self.device)
            l = l.to(self.device)

            self.optimizer.zero_grad()

            # Perf predictions for target, inferior, superior configurations.
            y_pred = self.model.forward(x)
            y_pred_s = self.model.forward(s)
            y_pred_l = self.model.forward(l) 

            output_gr_smaller = nn.Sigmoid()(y_pred - y_pred_s)
            larger_gr_output  = nn.Sigmoid()(y_pred_l - y_pred)
            larger_gr_smaller  = nn.Sigmoid()(y_pred_l - y_pred_s)

            logits = torch.cat([output_gr_smaller,larger_gr_output,larger_gr_smaller], 0)

            # Targets are all 1 implying target>smaller, larger>target, larger>smaller
            if not self.weighted:
                loss = nn.BCELoss()(logits, torch.ones_like(logits).to(self.device))
            else:
                loss = weighted_bpr_loss(logits, accuracies, ranks, self.weigh_fn, self.device)

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

        for (x, s, l), _, _ in self.mtrloader:

            x = x.to(self.device)
            s = s.to(self.device)
            l = l.to(self.device)
            
            self.optimizer.zero_grad()

            # Perf predictions for target, inferior, superior configurations. Range [0, 1]
            y_pred = self.model.forward(x) 
            y_pred_s = self.model.forward(s)
            y_pred_l = self.model.forward(l) 

            loss = nn.TripletMarginLoss(margin = margin)(y_pred, y_pred_l, y_pred_s)
            
            loss.backward()
            self.optimizer.step()

            trloss += float(loss)
            dlen+=1

        return trloss/dlen    

    def validation(self, _set):
        self.model.eval()
        self.model.to(self.device)
        self.mtrloader_unshuffled.dataset.set=_set 

        scores_5 = []
        scores_10 = []
        scores_20 = []
        ranks = []
        values = []
        predicted_y =[] 
        actual_y = []
        with torch.no_grad():
            for i,(x,acc,y_) in enumerate(self.mtrloader_unshuffled):
                x = x.to(self.device)
                y = acc.to(self.device).tolist()
                y_pred = self.model.forward(x).squeeze().tolist()

                actual_y += y
                predicted_y += y_pred

                if _set!="train": # batch_size is fixed
                    y_true = np.array(y).reshape(1,-1)
                    y_score = np.maximum(1e-7,np.array(y_pred)).reshape(1,-1)
                    scores_5.append(ndcg_score(y_true=y_true, y_score=y_score, k=5))
                    scores_10.append(ndcg_score(y_true=y_true, y_score=y_score, k=10))
                    scores_20.append(ndcg_score(y_true=y_true, y_score=y_score, k=20))
                    ranks.append(self.mtrloader_unshuffled.dataset.ranks[_set][i][np.argmax(y_pred)])
                    values.append(self.mtrloader_unshuffled.dataset.values[_set][i][np.argmax(y_pred)])         
            
            # Need to perform this with another loop due to possible sparsity
            if _set=="train":
                start = 0
                for i,ds in enumerate(np.unique(self.mtrloader_unshuffled.dataset.ds_ref)):
                    end = start + len(np.where(self.mtrloader_unshuffled.dataset.ds_ref==ds)[0])
                    y_true = np.array(actual_y[start:end]).reshape(1,-1)
                    y_score=np.maximum(1e-7,np.array(predicted_y[start:end])).reshape(1,-1)
                    y_pred = predicted_y[start:end]
                    scores_5.append(ndcg_score(y_true=y_true,y_score=y_score,k=5))
                    scores_10.append(ndcg_score(y_true=y_true,y_score=y_score,k=10))
                    scores_20.append(ndcg_score(y_true=y_true,y_score=y_score,k=20))
                    ranks.append(self.mtrloader_unshuffled.dataset.ranks[_set][i][np.argmax(y_pred)])
                    values.append(self.mtrloader_unshuffled.dataset.values[_set][i][np.argmax(y_pred)])
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
    parser.add_argument('--save_path', type=str, default='../../data/models/ZAP-HPO', 
                        help="The path of the model/log save directory")
    parser.add_argument('--data_path', type=str, default='../../data/meta_dataset', 
                        help="The path of the metadata directory")
    parser.add_argument('--config_path',type=str, default = "default_config.yaml",
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
    runner.train()

