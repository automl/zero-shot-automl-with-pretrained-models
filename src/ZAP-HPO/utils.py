#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import os
import yaml

class Log:
    def __init__(self, args, logf, summary=None):
        self.args = args
        self.logf = logf
        self.summary = summary
        self.stime = time.time()
        self.ep_sttime = None
    
    def print(self, logger, epoch, tag=None, avg=True):
        if tag == 'train':
            ct = time.time() - self.ep_sttime
            tt = time.time() - self.stime
            msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
            print(msg)
            self.logf.write(msg + '\n')
        logger.print_(header=tag, logfile=self.logf, avg=avg)
        
        if self.summary is not None:
            logger.add_scalars(
                self.summary, header=tag, step=epoch, avg=avg)
        logger.clear()
    
    def print_args(self, config=None):
        argdict = vars(self.args)
        if config is not None:
            argdict.update(config)
        print(argdict)
        for k, v in argdict.items():
            self.logf.write(k + ': ' + str(v) + '\n')
        self.logf.write('\n')
    
    def set_time(self):
        self.stime = time.time()
    
    def save_time_log(self):
        ct = time.time() - self.stime
        msg = f'({ct:6.2f}s) meta-training phase done'
        print(msg)
        self.logf.write(msg + '\n')
    
    def print_pred_log(self, loss, corr, tag, ndcg=None, epoch=None, max_corr_dict=None):
        if tag == 'train':
            ct = time.time() - self.ep_sttime
            tt = time.time() - self.stime
            msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
            self.logf.write(msg + '\n');
            print(msg);
            self.logf.flush()
        if max_corr_dict is not None:
            max_corr = max_corr_dict['rank@1']
            msg = f'{tag}: rank@1 {corr:.4f} ({max_corr:.4f})\n'
            msg+= f"NDCG@5 {ndcg['NDCG@5']:.4f} NDCG@10 {ndcg['NDCG@10']:.4f} NDCG@20 {ndcg['NDCG@20']:.4f}"
        else:
            msg = f'{tag}: loss {loss:.6f}'
        self.logf.write(msg + '\n');
        print(msg);
        self.logf.flush()
    
    def max_corr_log(self, max_corr_dict):
        corr = max_corr_dict['rank@1']
        loss = max_corr_dict['loss']
        epoch = max_corr_dict['epoch']
        msg = f'[epoch {epoch}] max correlation: {corr:.4f}, loss: {loss:.6f}'
        self.logf.write(msg + '\n');
        print(msg);
        self.logf.flush()
        
def construct_model_path(save_path, config_identifier, split_type, loo, cv, mode, weighted, weigh_fn, sparsity, use_meta):
    extra = f"-{sparsity}" if sparsity > 0 else ""
    extra += "-no-meta" if not use_meta else ""
    if mode == "regression" or mode == "bpr":
        extra += f"-function-{weigh_fn}" if weighted else ""

    if split_type!="loo":
        model_path = os.path.join(save_path, f"{'weighted-' if weighted else ''}{mode}{extra}", config_identifier, str(cv))
    else:
        model_path = os.path.join(save_path+"-cvplusloo", f"{'weighted-' if weighted else ''}{mode}{extra}", config_identifier, str(loo), str(cv))
    
    return model_path

def save_model(epoch, model, model_path, max_corr=None):
    print("==> save current model...")
    if max_corr is not None:
        torch.save(model.cpu().state_dict(),
                   os.path.join(model_path, 'ckpt_max_corr.pt'))
    else:
        torch.save(model.cpu().state_dict(),
                   os.path.join(model_path, f'ckpt_{epoch}.pt'))
        
def load_model(model, model_path,device):
    print("==> load best model...")
    model.load_state_dict(torch.load(os.path.join(model_path, 'ckpt_max_corr.pt'),map_location=device))
    
    
def get_log(epoch, loss, acc=None, ndcg5=None, ndcg10=None, ndcg20=None, tag='train', comment="valid"):
    if tag=="train":
        msg = f'[{tag}] Ep {epoch} loss {loss.item():0.4f} '
    else:
        msg = f'[{comment}] Ep {epoch} Top-1 Rank {loss} Top-1 Acc {acc} NDCG@5 {ndcg5:.4f} NDCG@10 {ndcg10:.4f} NDCG@20 {ndcg20:.4f}'    
    return msg


def config_from_yaml(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def config_to_yaml(config_path, config):
    with open(config_path, 'w') as stream:
        yaml.dump(config, stream)