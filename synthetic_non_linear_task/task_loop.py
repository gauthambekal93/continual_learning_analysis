# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:21:07 2026

@author: gauthambekal93
"""
import torch
from tqdm import tqdm
from models.feed_forward_network_2 import MLP
from task_generator import sample_task_params, sample_from_task
from utils.metrics import accuracy, grad_stats, total_avg_grad_norm, dead_relu_fraction, get_hessian_metrics, combine_hessian_metrics, get_param_signs, get_transition_probs, get_local_sensitivity
from utils.save_and_load import save_checkpoint
from training.train_loop import train_one_task


def run_tasks(   
    root_dir,   
    run_id,
    seed,    
    depth,
    width,
    num_tasks,
    epochs_per_task,
    batch_size,
    n_train,
    n_test,
    noise,
    lr,
    wd,
    activation,
    device,
    out_dir,
    save_every,
    print_every,
    residual ,
    norm,
    dropout,
    opt_name,
    momentum,
    return_pre_act,
    return_layer_out):
    
    rows = []
    
    net = MLP(width=width, depth=depth, activation=activation, residual = residual, norm = norm, dropout = dropout).to(device)
    
    if opt_name == 'adam':
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    
    if opt_name == 'sgd':
        opt = torch.optim.SGD( net.parameters(), lr=lr, momentum=momentum, weight_decay=wd )
    
    for t in tqdm(range(num_tasks), desc=f"seed={seed} depth={depth} width={width}"):
        # sample ONE task
        params = sample_task_params()
    
        # train and test from SAME task
        Xtr, ytr = sample_from_task(params, n=n_train, noise=noise)
        Xte, yte = sample_from_task(params, n=n_test, noise=noise)
        
        net.eval()
        acc_before = accuracy(net, Xte, yte, device=device)
        
        if (t) % save_every == 0:
            
            hessian_metrics_before = get_hessian_metrics( Xtr, ytr, device, net)
            param_signs_before = get_param_signs(net)
        
        net.train()
        
        avg_grad, loss_before, loss_after, avg_neg_count, avg_weight_size, avg_bias_size, avg_layer_weight_size, avg_layer_bias_size, avg_lop_score_layer = train_one_task(
            net,
            Xtr,
            ytr,
            opt,
            batch_size=batch_size,
            epochs_per_task=epochs_per_task,
            device=device
        )
        
        if t ==100:
            print("stop")
            print("stop")
            
        if (t) % save_every == 0:
            net.eval()    
            acc_after = accuracy(net, Xte, yte, device=device)
            acc_change = acc_after - acc_before
        
            #dead = dead_relu_fraction(net, Xte, device=device)
            avg_grad_norm = total_avg_grad_norm(avg_grad)
            hessian_metrics_after = get_hessian_metrics( Xtr, ytr, device, net)
            
            hessian_metrics = combine_hessian_metrics(hessian_metrics_before, hessian_metrics_after)
            
            result = net(Xtr, return_pre_act, return_layer_out)
            
            param_signs_after = get_param_signs(net)
            
            transition_probs = get_transition_probs(param_signs_before, param_signs_after)
            
            local_sensitivity = get_local_sensitivity(Xtr, device, net)
            
            row = {
                "task": t,
                "acc_before": acc_before,
                "acc_after": acc_after,
                "acc_change": acc_change,
                "loss_before":loss_before,
                "loss_after": loss_after,
                "avg_grad_norm": avg_grad_norm,
                "avg_weight_size":avg_weight_size,
                "avg_bias_size":avg_bias_size,
                "collapsed": int(acc_after < 0.55 and avg_grad_norm < 1e-6)
            }
            row.update(avg_layer_weight_size)
            row.update(avg_layer_bias_size)
            row.update(avg_grad)
            row.update(avg_neg_count)
            row.update(avg_lop_score_layer)
            
            if return_layer_out is True:
                row.update(result['dead_neuron_frac'])
                row.update(result['pos_neuron_frac'])
                row.update(result['mean_act'])
                row.update(result['prelu_param'])
                
            #row.update(dead)
            row.update(hessian_metrics)
            row.update(transition_probs)
            row.update(local_sensitivity)
            rows.append(row)
          
            print(    f"task={t} acc_before={acc_before:.3f} acc_after={acc_after:.3f} ")
        
            save_checkpoint(net, opt, rows, root_dir, out_dir, run_id, t + 1)
    
    save_checkpoint(net, opt, rows, root_dir, out_dir, run_id, num_tasks)
    
    
    