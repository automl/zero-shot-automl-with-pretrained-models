#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 07:30:28 2021

@author: hsjomaa
"""

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    from torch.autograd import Variable
except:
    raise ImportError("For this example you need to install pytorch.")

from sklearn.metrics import ndcg_score
import os
import json
import pickle
import numpy as np
import time
from tqdm import tqdm
from utils import Log,get_log,save_model
from loader import get_tr_loader,get_ts_loader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class batch_mlp(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu", dropout=0.0):
        
        super(batch_mlp, self).__init__()
        
        assert(nonlinearity=="relu")
        self.nonlinearity = nn.ReLU()
        
        self.fc = nn.ModuleList([nn.Linear(in_features=d_in, out_features=output_sizes[0])])
        for d_out in output_sizes[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))

        self.dropout = nn.Dropout(dropout)

        self.out_features = output_sizes[-1]
    
    def forward(self,x):
        
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        return x

def WMSE(input,target,weights):
    out = (input-target)**2
    out = out * weights
    return out.mean()

class ModelRunner:
    def __init__(self,args):

        self.args = args
        self.data_path = args.data_path
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.save_path = args.save_path
        self.loo =  args.loo
        self.cv = args.cv
        self.mode = args.mode
        self.seed = args.seed
        self.weighted = args.weighted
        self.sparsity = args.sparsity
        self.use_meta = args.use_meta
        self.fn = args.fn
        self.output_normalization = args.output_normalization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_corr_dict = {'rank@1': np.inf, 'epoch': -1, "ndcg@5":-1, "ndcg@10":-1, "ndcg@20":-1}
        cs = self.get_configspace(self.seed)
        config = cs.sample_configuration()

        self.model = batch_mlp(d_in=39 if self.use_meta else 35, output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1], dropout=config["dropout_rate"])
        self.model.to(self.device)
        
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
        elif config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'], weight_decay = config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch, eta_min= config["min_lr"])

        self.mtrloader,self.mtrloader_unshuffled =  get_tr_loader(config['batch_size'], self.data_path, loo=self.loo, cv=self.cv,
                                        mode=self.mode,split_type=args.split_type,sparsity =self.sparsity,
                                        use_meta=self.use_meta, output_normalization=self.output_normalization)
        
        self.mtrloader_test =  get_ts_loader(525, self.data_path, self.loo,
                                              mu_in=self.mtrloader.dataset.mean_input,
                                              std_in=self.mtrloader.dataset.std_input,
                                              mu_out=self.mtrloader.dataset.mean_output,
                                              std_out=self.mtrloader.dataset.std_output,split_type=args.split_type,
                                              use_meta=self.use_meta)
        
        self.acc_mean = self.mtrloader.dataset.mean_output
        self.acc_std = self.mtrloader.dataset.std_output
        extra = f"-{self.sparsity}" if self.sparsity > 0 else ""
        extra += "-no-meta" if not self.use_meta else ""
        extra += "-normalized" if not self.output_normalization else ""
        if self.mode == "bpr":
            extra += f"-function-{self.fn}" if self.weighted else ""

        if args.split_type!="loo":
            self.model_path = os.path.join(self.save_path, f"{'weighted-' if self.weighted else ''}{self.mode}{extra}", str(self.seed), str(self.cv))
        else:
            self.model_path = os.path.join(self.save_path+"-cvplusloo", f"{'weighted-' if self.weighted else ''}{self.mode}{extra}", str(self.seed), str(self.loo),str(self.cv))
        
        os.makedirs(self.model_path,exist_ok=True)
        self.mtrlog = Log(self.args, open(os.path.join(self.model_path, 'meta_train_predictor.log'), 'w'))
        self.mtrlog.print_args(config)    

        norm_dict = {'mean': self.mtrloader.dataset.mean_input, 'std': self.mtrloader.dataset.std_input}

        norm_path = os.path.join(self.model_path, "norm.pt")      
        with open(norm_path, 'wb') as f:
            pickle.dump(norm_dict, f) 
    
    def train(self):
        history = {"trndcg": [], "vandcg": []}
        patience = 0
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            if self.mode=="regression":
                loss = self.train_epoch(epoch)  
            elif self.mode=="bpr":
                loss = self.train_bpr_epoch(epoch)
            elif self.mode=="tml":
                loss = self.train_tml_epoch(epoch)

            patience += 1

            self.scheduler.step()

            self.mtrlog.print_pred_log(loss, 0, 'train', epoch=epoch)
            vacorr, vaccc, vandcg = self.validation(epoch, "valid")
            trcorr, tracc, trndcg = self.validation(epoch, "train")
            tecorr, teacc, tendcg = self.test(epoch)

            if self.max_corr_dict['rank@1'] >= vacorr:
                patience = 0

            if self.max_corr_dict['rank@1'] > vacorr:
                self.max_corr_dict['rank@1'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict.update(vandcg)
                save_model(epoch, self.model, self.model_path, max_corr=True)
                
            #if patience >= 25:
             #   break

            self.mtrlog.print_pred_log(0, vacorr, 'valid', ndcg=vandcg, max_corr_dict=self.max_corr_dict)
            
            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)
            
            vandcg.update({"acc":vaccc,
                           "rank":vacorr})                          
            trndcg.update({"acc":tracc,
                           "rank":trcorr,
                           "loss":loss})              
            tendcg.update({"acc":teacc,
                           "rank":tecorr,
                           })

            history["trndcg"].append(trndcg)
            history["vandcg"].append(vandcg)  

            history_path = os.path.join(self.model_path, "history.json")      
            with open(history_path, 'w') as f:
                json.dump(history, f)  

        self.mtrlog.save_time_log()

        return history
        
    def train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()
        self.mtrloader.dataset.training="train"
        dlen = 0
        trloss = 0
        pbar = self.mtrloader
        for x, acc, y_ in pbar:
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            y = acc.to(self.device)
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

    def train_bpr_epoch(self, epoch, int_normalization = True):
        self.model.to(self.device)
        self.model.train()
        dlen = 0
        trloss = 0
        pbar = self.mtrloader

        for (x, s, l), (acc,acc_s,acc_l), (r,r_s,r_l) in pbar:
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            y_pred_s = self.model.forward(s)
            y_pred_l = self.model.forward(l) 


            output_gr_smaller = nn.Sigmoid()(y_pred - y_pred_s) # batch
            larger_gr_output  = nn.Sigmoid()(y_pred_l - y_pred)
            larger_gr_smaller  = nn.Sigmoid()(y_pred_l - y_pred_s)

            logits = torch.cat([output_gr_smaller,larger_gr_output,larger_gr_smaller], 0) # concatenates end to end

            if self.weighted:
                if self.fn=="v0":
                    weights = torch.cat([torch.exp(-(acc-acc_s).pow(2)),
                                        torch.exp(-(acc_l-acc).pow(2)),
                                        torch.exp(-(acc_l-acc_s).pow(2))],0)
                elif self.fn=="v1":
                    weights = torch.cat([(acc-acc_s).pow(2),
                                        (acc_l-acc).pow(2),
                                        (acc_l-acc_s).pow(2)],0)
                elif self.fn=="v0-rank":
                    weights = torch.cat([torch.exp(-((r-r_s)).pow(2)),
                                        torch.exp(-((r_l-r)).pow(2)),
                                        torch.exp(-((r_l-r_s)).pow(2))],0)
                elif self.fn=="v1-rank":
                    weights = torch.cat([((r-r_s)).pow(2),
                                        ((r_l-r)).pow(2),
                                        ((r_l-r_s)).pow(2)],0)

                loss = nn.BCELoss(weight=weights.unsqueeze(-1))(logits, torch.ones_like(logits))    
            else:
                loss = nn.BCELoss()(logits, torch.ones_like(logits))

            loss.backward()
            self.optimizer.step()

            trloss += float(loss)
            dlen+=1

        return trloss/dlen

    def train_tml_epoch(self, epoch, margin = 1.0):
        self.model.to(self.device)
        self.model.train()
        dlen = 0
        trloss = 0
        pbar = self.mtrloader

        for (x, s, l), (acc,acc_s,acc_l), (r,r_s,r_l) in pbar:
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

    def validation(self, epoch, training):
        self.model.to(self.device)
        self.model.eval()
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
            y_pred = self.model.forward(x).squeeze().tolist()
            y = acc.to(self.device).tolist()
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

    def test(self, epoch):
        self.model.to(self.device)
        self.model.eval()
        pbar = self.mtrloader_test
        scores_5 = []
        scores_10 = []
        scores_20 = []
        ranks = []
        values = []
        with torch.no_grad():
          for i,(x,acc) in enumerate(pbar):
            y_pred = self.model.forward(x)
            y = acc.to(self.device)
            y = y.tolist()
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
        min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-10, upper=1e-6, log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD', 'AdamW']) # added SGD AdamW
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-2, log=True) # added
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=8, upper=128, default_value=64) # added
        cs.add_hyperparameters([lr, min_lr, optimizer, weight_decay, batch_size])
        
        num_hidden_layers =  CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=2, upper=10)
        num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=32, upper=512) # same
        cs.add_hyperparameters([num_hidden_layers, num_hidden_units])
        
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, log=False)
        cs.add_hyperparameters([dropout_rate])
        
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, log=False) # added for SGD
        cs.add_hyperparameters([sgd_momentum])
        momentum_cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_conditions([momentum_cond])
        
        return cs    
    
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='../ckpts_norm', help='the path of save directory')
    parser.add_argument('--data_path', type=str, default='../../data', help='the path of save directory')
    parser.add_argument('--mode', type=str, default='bpr', help='training objective',choices=["regression", "bpr", "tml"])
    parser.add_argument('--save_epoch', type=int, default=20, help='how many epochs to wait each time to save model states') 
    parser.add_argument('--max_epoch', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--loo', type=int, default=1, help='Index of dataset [0,34] that should be removed')
    parser.add_argument('--cv', type=int, default=1, help='Index of CV [1,5]')
    parser.add_argument('--split_type', type=str, default="cv", help='cv|loo')
    parser.add_argument('--weighted', type=str, default="False", choices=["True","False"])
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--use_meta', type=str, default="True", choices=["True","False"])    
    parser.add_argument('--output_normalization', type=str, default="True", choices=["True","False"])
    parser.add_argument('--fn', type=str, default="v0", choices=["v0","v1", "v0-rank", "v1-rank"])
    args = parser.parse_args()
    args.weighted = eval(args.weighted)
    args.use_meta = eval(args.use_meta)
    args.output_normalization = eval(args.output_normalization)

    runner = ModelRunner(args)
    history = runner.train()

    history_path = os.path.join(runner.model_path, "history.json")      
    with open(history_path, 'w') as f:
        json.dump(history, f)

