# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:01:28 2026

@author: gauthambekal93
"""


    
import torch
import numpy as np
from torch import nn

# ---------------- metrics ----------------
@torch.no_grad()
def accuracy(net, X, y, device="cpu"):
    net.eval()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    pred = net(X).argmax(dim=1)
    return (pred == y).float().mean().item()


def grad_stats(net):
    out = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            out[name + "_grad_norm"] = p.grad.detach().norm().item()
            out[name + "_weight_norm"] = p.detach().norm().item()
    return out


def total_avg_grad_norm(avg_grad):
    vals = [v for k, v in avg_grad.items() if "grad_norm" in k]
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


@torch.no_grad()
def dead_relu_fraction(net, X, device="cpu"):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    z = X
    out = {}
    relu_id = 0

    for layer in net.net:
        z = layer(z)
        if isinstance(layer, nn.ReLU):
            out[f"relu_{relu_id}_dead_frac"] = (z <= 0).float().mean().item()
            relu_id += 1

    return out