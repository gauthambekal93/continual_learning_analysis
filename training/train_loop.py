# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:03:08 2026

@author: gauthambekal93
"""

import torch
from torch import nn
from utils.metrics import grad_stats
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

    total_grad = {}
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
            loss = ce(net(xb), yb)
            loss.backward()
 
            batch_stats = grad_stats(net)
            for k, v in batch_stats.items():
                if "grad_norm" in k:
                    total_grad[k] = total_grad.get(k, 0.0) + v

            opt.step()
            num_batches += 1
            
            if loss_before is None:
                loss_before = loss.item()
                
    loss_after = loss.item()
    
    avg_grad = {
        k: v / max(num_batches, 1)
        for k, v in total_grad.items()
    }

    return avg_grad, loss_before, loss_after
