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

'''
def grad_stats(net):
    out = {}
    weight_size, param_count = 0, 0
    for name, p in net.named_parameters():
        if p.grad is not None:
            out[name + "_grad_norm"] = p.grad.detach().norm().item()
            out[name + "_weight_norm"] = p.detach().norm().item()
            out[name + "_percent_neg_weight"] = p[p<=0].numel() / p.numel()
            weight_size = weight_size + p.sum().item()
            param_count = param_count + len(p)
            
            if  out[name + "_percent_neg_weight"] >1:
                print("stop")
                print("stop")
                
    normalized_weight_size = weight_size / param_count
    
    return out, weight_size, normalized_weight_size
'''

def grad_stats(net):
    out = {}
    total_weight_size, total_bias_size, weight_count, bias_count = 0, 0, 0, 0
    
    for name, p in net.named_parameters():
        if p.grad is not None:
            out[name + "_grad_norm"] = p.grad.detach().norm().item()
            out[name + "_weight_norm"] = p.detach().norm().item()
            out[name + "_percent_neg_weight"] = p[p<=0].numel() / p.numel()
            
            if 'weight' in name:
                total_weight_size = total_weight_size + p.sum().item()
                weight_count = weight_count + p.numel()
            if 'bias' in name:
                total_bias_size = total_bias_size + p.sum().item()
                bias_count = bias_count + p.numel()
            
    avg_weight_size = total_weight_size / weight_count
    
    avg_bias_size = total_bias_size / bias_count
    
    return out, avg_weight_size, avg_bias_size


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



def get_hessian_metrics(X, y, device, net):

    ce = nn.CrossEntropyLoss()
    net.train()

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)

    loss = ce(net(X), y)

    params = [p for p in net.parameters() if p.requires_grad]

    # first-order gradients
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True
    )

    # build full Hessian matrix
    hessian_rows = []

    for grad in grads:

        grad_flat = grad.reshape(-1)

        for g in grad_flat:

            second_grads = torch.autograd.grad(
                g,
                params,
                retain_graph=True
            )

            row = torch.cat([
                sg.reshape(-1)
                for sg in second_grads
            ])

            hessian_rows.append(row)

    # [N, N]
    hessian = torch.stack(hessian_rows)
     
    # force symmetric (theoretically hessian is symmetric, but pytorch can have slight variation due to float precision etc, so forcing the symmetry to be exact)
    hessian = 0.5 * (hessian + hessian.T)
    
    # numerical stabilization
    eps = 1e-6
    hessian = hessian + eps * torch.eye(
        hessian.shape[0],
        device=hessian.device
    )


    # eigenvalues
    eigvals_raw = torch.linalg.eigvalsh(hessian)

    eigvals_pos = torch.clamp(eigvals_raw, min=0)
    eigvals_abs = eigvals_raw.abs()

    # traces
    trace_signed = eigvals_raw.sum()
    trace_pos = eigvals_pos.sum()
    trace_abs = eigvals_abs.sum()

    # extrema
    min_eigenval = eigvals_raw.min()
    max_eigenval = eigvals_raw.max()

    # negatives
    num_negative = (eigvals_raw < 0).sum()

    # effective rank + participation ratio (positive)
    p_pos = eigvals_pos / (eigvals_pos.sum() + 1e-12)

    effective_rank_pos = torch.exp(
        -(p_pos * torch.log(p_pos + 1e-12)).sum()
    )

    participation_ratio_pos = (
        (eigvals_pos.sum() ** 2)
        /
        (torch.sum(eigvals_pos ** 2) + 1e-12)
    )

    # effective rank + participation ratio (absolute)
    p_abs = eigvals_abs / (eigvals_abs.sum() + 1e-12)

    effective_rank_abs = torch.exp(
        -(p_abs * torch.log(p_abs + 1e-12)).sum()
    )

    participation_ratio_abs = (
        (eigvals_abs.sum() ** 2)
        /
        (torch.sum(eigvals_abs ** 2) + 1e-12)
    )

    # counts
    num_large_pos_eigs = (eigvals_pos > 1e-4).sum()
    num_large_abs_eigs = (eigvals_abs > 1e-4).sum()

    return {

        "trace_signed": trace_signed.item(),
        "trace_pos": trace_pos.item(),
        "trace_abs": trace_abs.item(),

        "min_eigenval": min_eigenval.item(),
        "max_eigenval": max_eigenval.item(),

        "num_negative": num_negative.item(),

        "effective_rank_pos": effective_rank_pos.item(),
        "effective_rank_abs": effective_rank_abs.item(),
        
        "participation_ratio_pos": participation_ratio_pos.item(),
        "participation_ratio_abs": participation_ratio_abs.item(),

        "num_large_pos_eigs": num_large_pos_eigs.item(),
        "num_large_abs_eigs": num_large_abs_eigs.item()
    }


def combine_hessian_metrics(hessian_metrics_before, hessian_metrics_after):
    hessian_metrics = {}
    for (k1, v1), (k2, v2) in zip(hessian_metrics_before.items(), hessian_metrics_after.items()):
        hessian_metrics[k1+'_before']=v1
        hessian_metrics[k2+'_after']=v2
    
    return hessian_metrics

