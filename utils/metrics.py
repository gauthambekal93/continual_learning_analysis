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
    result = net(X)
    pred = result["pred"].argmax(dim=1)
    return (pred == y).float().mean().item()



def get_fully_nonpos_ip_neuron_percent(net):
    layer_num = 0
    fully_nonpos_ip_neuron_percent = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
     
            
            weight = module.weight
            bias = module.bias
            
            weight_bool = weight<=0
            bias_bool = (bias<=0).reshape(-1, 1)
            combined_bool = torch.cat([weight_bool, bias_bool], dim =1)
            
            fully_nonpos_ip_neuron_percent['fully_nonpos_ip_neuron_percent_layer_'+str(layer_num) ] =  100* (combined_bool.all(dim = 1).sum().item() / combined_bool.shape[0] )
            layer_num += 1
    return  fully_nonpos_ip_neuron_percent   




def get_layer_lop_score(net, eps=1e-12):
    layer_num = 0
    layer_lop_score = {}

    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            bias = module.bias

            weight_bool = weight <= 0
            bias_bool = (bias <= 0).reshape(-1, 1)
            combined_bool = torch.cat([weight_bool, bias_bool], dim=1)

            nonpos_count = combined_bool.sum(dim=1).float()

            grad_available_prob = 1.0 / (2.0 ** nonpos_count)

            lop_score = (-torch.log(grad_available_prob + eps)).mean().item()

            layer_lop_score[f"lop_score_layer_{layer_num}"] = lop_score
            layer_num += 1

    return layer_lop_score



def get_layer_lop_score_2(net, eps=1e-12):
    layer_num = 0
    layer_lop_score = {}

    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            bias = module.bias

            weight_bool = weight <= 0
            bias_bool = (bias <= 0).reshape(-1, 1)
            combined_bool = torch.cat([weight_bool, bias_bool], dim=1)

            nonpos_count = combined_bool.sum(dim=1).float()

            grad_available_prob = 1.0 / (2.0 ** nonpos_count)
            
            lop_score = (1.0 / (2.0 ** nonpos_count) ).mean().item()
            #lop_score = (-torch.log(grad_available_prob + eps)).mean().item()

            layer_lop_score[f"lop_score_layer_{layer_num}"] = lop_score
            layer_num += 1
    
    return layer_lop_score


def grad_stats(net):
    out = {}
    total_weight_size, total_bias_size, weight_count, bias_count = 0, 0, 0, 0
    
    for name, p in net.named_parameters():
        if p.grad is not None:
            out[name + "_grad_norm"] = p.grad.detach().norm().item()
            out[name + "_weight_norm"] = p.detach().norm().item()
            out[name + "_percent_neg_weight"] = p[p<=0].numel() / p.numel()
            
            if 'weight' in name:
                out[name + "_size"] = p.sum().item() / p.numel()
                total_weight_size = total_weight_size + p.sum().item()
                weight_count = weight_count + p.numel()
            
            if 'bias' in name:
                out[name + "_size"] = p.sum().item() / p.numel()
                total_bias_size = total_bias_size + p.sum().item()
                bias_count = bias_count + p.numel()
            
    avg_weight_size = total_weight_size / weight_count
    
    avg_bias_size = total_bias_size / bias_count
    
    #layer_lop_score = get_layer_lop_score(net)
    
    layer_lop_score = get_layer_lop_score_2(net)
    
    out.update(layer_lop_score)
    
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
    pred = net(X)["pred"]
    
    loss = ce(pred, y)

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





def get_param_signs(net):
     
    param_signs = {}

    for name, param in net.named_parameters():
        
        p = param.detach().cpu()
        
        if 'weight' in name.lower() and p.ndim == 1:  # We dont want PRELU activation weight. 
            continue
            
        if p.ndim == 1:          # bias
            for i in range(p.shape[0]):
    
                uid = f"{name}[{i}]"
    
                param_signs[uid] = int(p[i] > 0)
    
        elif p.ndim == 2:        # weight
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
    
                    uid = f"{name}[{i},{j}]"
    
                    param_signs[uid] = int(p[i,j] > 0)
                    
                    
    return param_signs
                
                   
                
def get_transition_probs(param_signs_before, param_signs_after):
    #pos_to_pos = 0
    #pos_to_neg = 0
    #neg_to_pos = 0
    #neg_to_neg = 0
    #total = 0
    
    transition_probs = {}
    total = {}             
    
    for k in param_signs_before.keys():
        if 'weight' in k:
            layer_name = k.split('weight')[0]
            
        elif 'bias' in k:
            layer_name = k.split('bias')[0]
        
        for transition_name in ["pos_to_pos", "pos_to_neg", "neg_to_pos", "neg_to_neg"]:
            transition_probs[layer_name + transition_name] = 0
            
        total[layer_name] = 0
        
    for (k1, v1), (k2, v2) in zip(param_signs_before.items(), param_signs_after.items()):
        
        if 'weight' in k1:
            layer_name = k1.split('weight')[0]
            
        elif 'bias' in k1:
            layer_name = k1.split('bias')[0]
            
        if v1 ==0 and v2==0:
            transition_probs[layer_name + 'neg_to_neg'] = transition_probs[layer_name + 'neg_to_neg'] + 1
            #neg_to_neg = neg_to_neg + 1
        if v1==0 and v2==1:
            transition_probs[layer_name + 'neg_to_pos'] = transition_probs[layer_name + 'neg_to_pos'] + 1
            #neg_to_pos = neg_to_pos + 1
        if v1==1 and v2==0:
            transition_probs[layer_name + 'pos_to_neg'] = transition_probs[layer_name + 'pos_to_neg'] + 1
            #pos_to_neg = pos_to_neg + 1
        if v1==1 and v2==1:
            transition_probs[layer_name + 'pos_to_pos'] = transition_probs[layer_name + 'pos_to_pos'] + 1
            #pos_to_pos = pos_to_pos + 1
        
        total[layer_name] = total[layer_name] + 1    
        #total = total + 1
    for layer_name in total.keys():
        transition_probs[layer_name + 'neg_to_neg'] = transition_probs[layer_name + 'neg_to_neg'] / total[layer_name]
        transition_probs[layer_name + 'neg_to_pos'] = transition_probs[layer_name + 'neg_to_pos'] / total[layer_name]
        transition_probs[layer_name + 'pos_to_neg'] = transition_probs[layer_name + 'pos_to_neg'] / total[layer_name]
        transition_probs[layer_name + 'pos_to_pos'] = transition_probs[layer_name + 'pos_to_pos'] / total[layer_name]

    return transition_probs
    
    #pos_to_pos, pos_to_neg, neg_to_pos, neg_to_neg = pos_to_pos/ total, pos_to_neg / total, neg_to_pos/ total, neg_to_neg/ total
    
    #return { "pos_to_pos": pos_to_pos, 
    #         "pos_to_neg": pos_to_neg, 
    #         "neg_to_pos": neg_to_pos, 
    #         "neg_to_neg": neg_to_neg 
    #         }

 



    
 
        
    
    
     



