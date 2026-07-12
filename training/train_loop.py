# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:03:08 2026

@author: gauthambekal93
"""

import torch
from torch import nn
from utils.metrics import grad_stats
import numpy as np

def train_one_task(
    net,
    X,
    y,
    opt,
    batch_size=64,
    epochs_per_task=5,
    device="cpu"
):
    ce = nn.CrossEntropyLoss()
    net.train()

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)

    total_grad, neg_weight_count = {}, {}
    total_weight_size, total_bias_size = [], []
    total_layer_weight_size, total_layer_bias_size = {}, {}
    #fully_nonpos_ip_neuron_percent ={}
    lop_score_layer ={}
    
    num_batches = 0
    loss_before, loss_after = None, None

    for epoch in range(epochs_per_task):
        idx = torch.randperm(len(X), device=device)
        X_epoch = X[idx]
        y_epoch = y[idx]

        for i in range(0, len(X_epoch), batch_size):
            xb = X_epoch[i:i + batch_size]
            yb = y_epoch[i:i + batch_size]

            opt.zero_grad()
            result = net(xb)
            
            pred = result["pred"]
            loss = ce(pred, yb)
            loss.backward()
            
            #batch_stats, weight_size, normalized_weight_size = grad_stats(net)
            batch_stats, weight_size, bias_size = grad_stats(net)
        
            
            total_weight_size.append(weight_size)
            total_bias_size.append(bias_size)
            
            for k, v in batch_stats.items():
                if "grad_norm" in k:
                    total_grad[k] = total_grad.get(k, 0.0) + v
                if "percent_neg_weight" in k:    
                    neg_weight_count[k] =  neg_weight_count.get(k, 0.0) + v
                
                if 'weight_size' in k:    
                    total_layer_weight_size[k] = total_layer_weight_size.get(k, 0.0) + v
                
                if 'bias_size' in k:
                    total_layer_bias_size[k] = total_layer_bias_size.get(k, 0.0) + v
                
                if 'lop_score_layer' in k:
                    lop_score_layer[k] = lop_score_layer.get(k, 0.0) + v
                
            opt.step()
            num_batches += 1
            
            if loss_before is None:
                loss_before = loss.item()
                
    loss_after = loss.item()
    
    avg_weight_size = np.mean( total_weight_size )
    avg_bias_size = np.mean(total_bias_size)
    
    avg_grad = { k: v / max(num_batches, 1) for k, v in total_grad.items() }
   
    avg_neg_count = { k: v / max(num_batches, 1) for k, v in neg_weight_count.items() }

    avg_layer_weight_size =  { k: v / max(num_batches, 1) for k, v in total_layer_weight_size.items() }
    
    avg_layer_bias_size =  { k: v / max(num_batches, 1) for k, v in total_layer_bias_size.items() }
    
    avg_lop_score_layer = { k: v / max(num_batches, 1) for k, v in lop_score_layer.items() }
    
    return avg_grad, loss_before, loss_after, avg_neg_count, avg_weight_size, avg_bias_size, avg_layer_weight_size, avg_layer_bias_size, avg_lop_score_layer




