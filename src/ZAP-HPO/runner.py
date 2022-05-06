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
except:
	raise ImportError("For this example you need to install pytorch.")

from sklearn.metrics import ndcg_score
import os
import numpy as np
import time
from tqdm import tqdm
from utils import Log, get_log, save_model
from loader import get_tr_loader, get_ts_loader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
#from torch.utils.tensorboard import SummaryWriter


class batch_mlp(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu",dropout=0.0):
        
        super(batch_mlp, self).__init__()
        assert(nonlinearity=="relu")
        self.nonlinearity = nn.ReLU()
        self.fc = nn.ModuleList([nn.Linear(in_features=d_in, out_features=output_sizes[0])])
        for d_out in output_sizes[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))
        self.out_features = output_sizes[-1]
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        return x

class ModelRunner:
    def __init__(self,args):

        self.args = args
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.save_path = args.save_path
        self.loo =  args.loo
        self.cv = args.cv
        self.mode = args.mode
        self.seed = args.seed
        self.sparsity = args.sparsity
        self.use_meta = args.use_meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_corr_dict = {'rank@1': np.inf, 'epoch': -1, "ndcg@5":-1, "ndcg@10":-1, "ndcg@20":-1}
        cs = self.get_configspace(self.seed)
        config = cs.sample_configuration()
        self.model = batch_mlp(d_in=42 if self.use_meta else 38,output_sizes=config["num_hidden_layers"]*[config["num_hidden_units"]]+[1],
                               dropout=config["dropout_rate"])
        self.model.to(self.device)
        config["lr"]=1e-3
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])        
        self.criterion = nn.MSELoss()
        self.mtrloader,self.mtrloader_unshuffled =  get_tr_loader(self.batch_size, self.data_path, loo=self.loo, cv=self.cv,
                                        mode=self.mode,split_type=args.split_type,sparsity =self.sparsity,
                                        use_meta=self.use_meta)
        self.mtrloader_test =  get_ts_loader(525, self.data_path, self.loo,
                                              mu_in=self.mtrloader.dataset.mean_input,
                                              std_in=self.mtrloader.dataset.std_input,
                                              mu_out=self.mtrloader.dataset.mean_output,
                                              std_out=self.mtrloader.dataset.std_output,split_type=args.split_type,
                                              use_meta=self.use_meta)
        
        self.acc_mean = self.mtrloader.dataset.mean_output
        self.acc_std = self.mtrloader.dataset.std_output
        extra = str(-self.sparsity) if self.sparsity > 0 else ""
        extra += "-no-meta" if not self.use_meta else ""
        self.model_path = os.path.join(self.save_path, str(self.mode)+extra, str(self.cv))
        
        os.makedirs(self.model_path, exist_ok=True)
        self.mtrlog = Log(self.args, open(os.path.join(self.model_path, 'meta_train_predictor.log'), 'w'))
        self.mtrlog.print_args(config)
        #self.setup_writers()
        
    def setup_writers(self,):
        train_log_dir = os.path.join(self.model_path,"train")
        os.makedirs(train_log_dir,exist_ok=True)
        self.train_summary_writer = SummaryWriter(train_log_dir)
        
        valid_log_dir = os.path.join(self.model_path,"valid")
        os.makedirs(valid_log_dir,exist_ok=True)
        self.valid_summary_writer = SummaryWriter(valid_log_dir)     
        
        test_log_dir = os.path.join(self.model_path,"test")
        os.makedirs(test_log_dir,exist_ok=True)
        self.test_summary_writer = SummaryWriter(test_log_dir)       
    
    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            if self.mode=="regression":
                loss = self.train_epoch(epoch)  
            else:
                loss = self.train_bpr_epoch(epoch)
            # self.scheduler.step(loss)
            self.mtrlog.print_pred_log(loss, 0, 'train', epoch=epoch)
            vacorr, vaccc, vandcg = self.validation(epoch, "valid")
            trcorr, tracc, trndcg = self.validation(epoch, "train")
            tecorr, teacc, tendcg = self.test(epoch)
            if self.max_corr_dict['rank@1'] > vacorr:
                self.max_corr_dict['rank@1'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict.update(vandcg)
                save_model(epoch, self.model, self.model_path, max_corr=True)
    
            self.mtrlog.print_pred_log(0, vacorr, 'valid',ndcg=vandcg, max_corr_dict=self.max_corr_dict)
            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)
            vandcg.update({"acc":vaccc,
                           "rank":vacorr})
            
            #for k,v in vandcg.items():
                #self.valid_summary_writer.add_scalar(k, v, epoch)                            
            trndcg.update({"acc":tracc,
                           "rank":trcorr,
                           "loss":loss})
            #for k,v in trndcg.items():
                #self.train_summary_writer.add_scalar(k, v, epoch)                
            tendcg.update({"acc":teacc,
                           "rank":tecorr,
                           })
            #for k,v in tendcg.items():
                #self.test_summary_writer.add_scalar(k, v, epoch)                    
        self.mtrlog.save_time_log()
        
    def train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()
        self.mtrloader.dataset.training="train"
        dlen = 0
        trloss = 0
        pbar = tqdm(self.mtrloader)
    
        for x, acc, y_ in pbar:
          self.optimizer.zero_grad()
          y_pred = self.model.forward(x)
          y = acc.to(self.device)
          loss = self.criterion(y_pred, y.unsqueeze(-1))
          loss.backward()
          self.optimizer.step()
    
          y = y.tolist()
          y_pred = y_pred.squeeze().tolist()
          pbar.set_description(get_log(epoch, loss))
          trloss += float(loss)
          dlen+=1
    
        return trloss/dlen

    def train_bpr_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()
        dlen = 0
        trloss = 0
        pbar = tqdm(self.mtrloader)
    
        for (x, s, l), (acc,acc_s,acc_l), (r,r_s,r_l) in pbar:
          self.optimizer.zero_grad()
          y_pred = self.model.forward(x)
          y_pred_s = self.model.forward(s)
          y_pred_l = self.model.forward(l)

          output_gr_smaller = nn.Sigmoid()(y_pred - y_pred_s) 
          larger_gr_output  = nn.Sigmoid()(y_pred_l - y_pred) 
          larger_gr_smaller  = nn.Sigmoid()(y_pred_l - y_pred_s)
          logits = torch.cat([output_gr_smaller,larger_gr_output,larger_gr_smaller],0)
          loss = nn.BCELoss()(logits,torch.ones_like(logits))
          loss.backward()
          self.optimizer.step()
          
          pbar.set_description(get_log(epoch, loss))
          trloss += float(loss)
          dlen+=1
        return trloss/dlen

    def validation(self, epoch, training):
        self.model.to(self.device)
        self.model.eval()
        self.mtrloader_unshuffled.dataset.training=training
        pbar = tqdm(self.mtrloader_unshuffled)
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
        pbar.set_description(get_log(epoch, np.mean(ranks),
                                      acc = np.mean(values),
                                      ndcg5=np.mean(scores_5),
                                      ndcg10=np.mean(scores_10),
                                      ndcg20=np.mean(scores_20),
                                      tag="val",comment=training))
        return np.mean(ranks),np.mean(values), {"NDCG@5":np.mean(scores_5),"NDCG@10":np.mean(scores_10),"NDCG@20":np.mean(scores_20)}

    def test(self, epoch):
        self.model.to(self.device)
        self.model.eval()
        pbar = tqdm(self.mtrloader_test)
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
        pbar.set_description(get_log(epoch, np.mean(ranks),
                                      acc = np.mean(values),
                                      ndcg5=np.mean(scores_5),
                                      ndcg10=np.mean(scores_10),
                                      ndcg20=np.mean(scores_20),
                                      tag="val",comment="test"))            
        return np.mean(ranks),np.mean(values), {"NDCG@5":np.mean(scores_5),"NDCG@10":np.mean(scores_10),"NDCG@20":np.mean(scores_20)}


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
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--save_path', type=str, default='../ckpts', help='the path of save directory')
    parser.add_argument('--data_path', type=str, default='../../data', help='the path of save directory')
    parser.add_argument('--mode', type=str, default='bpr', help='training objective',choices=["regression","bpr"])
    parser.add_argument('--save_epoch', type=int, default=20, help='how many epochs to wait each time to save model states') 
    parser.add_argument('--max_epoch', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for generator')
    parser.add_argument('--loo', type=int, default=1, help='Index of dataset [0,34] that should be removed')
    parser.add_argument('--cv', type=int, default=1, help='Index of CV [1,5]')
    parser.add_argument('--split_type', type=str, default="cv", help='cv|loo')
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--use_meta', type=str, default="True", choices=["True","False"])    
    args = parser.parse_args()
    args.use_meta = eval(args.use_meta)
    runner = ModelRunner(args)
    runner.train()
