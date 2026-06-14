# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:21:07 2026

@author: gauthambekal93
"""
import torch
from tqdm import tqdm
from models.feed_forward_network import MLP
from task_generator import sample_task_params, sample_from_task
from utils.metrics import accuracy, grad_stats, total_avg_grad_norm, dead_relu_fraction
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
    norm):
    
    rows = []
    
    net = MLP(width=width, depth=depth, activation=activation, residual = residual, norm = norm).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    
    for t in tqdm(range(num_tasks), desc=f"seed={seed} depth={depth} width={width}"):
        # sample ONE task
        params = sample_task_params()
    
        # train and test from SAME task
        Xtr, ytr = sample_from_task(params, n=n_train, noise=noise)
        Xte, yte = sample_from_task(params, n=n_test, noise=noise)
    
        acc_before = accuracy(net, Xte, yte, device=device)
    
        avg_grad, loss_before, loss_after = train_one_task(
            net,
            Xtr,
            ytr,
            opt,
            batch_size=batch_size,
            epochs_per_task=epochs_per_task,
            device=device
        )
    
        acc_after = accuracy(net, Xte, yte, device=device)
        acc_change = acc_after - acc_before
    
        dead = dead_relu_fraction(net, Xte, device=device)
        avg_grad_norm = total_avg_grad_norm(avg_grad)
    
        row = {
            "run_id": run_id, 
            "seed": seed,
            "task": t,
            "depth": depth,
            "width": width,
            "activation": activation,
            "epochs_per_task": epochs_per_task,
            "n_train": n_train,
            "n_test": n_test,
            "noise": noise,
            "lr": lr,
            "wd": wd,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "acc_change": acc_change,
            "loss_before":loss_before,
            "loss_after": loss_after,
            "avg_grad_norm": avg_grad_norm,
            "collapsed": int(acc_after < 0.55 and avg_grad_norm < 1e-6),
        }
    
        row.update(avg_grad)
        row.update(dead)
        rows.append(row)
    
    
        print(
            f"seed={seed} epochs_per_task={epochs_per_task} depth={depth} width={width} task={t} "
            f"loss_before={loss_before:.3f} loss_after={loss_after:.3f} "
            f"acc_before={acc_before:.3f} acc_after={acc_after:.3f} "
            f"acc_change={acc_change:.3f}"
            f"grad={avg_grad_norm:.3e}"
        )
    
        if (t + 1) % save_every == 0:
       
            save_checkpoint(net, opt, rows, root_dir, out_dir, run_id, t + 1)
    
    save_checkpoint(net, opt, rows, root_dir, out_dir, run_id, num_tasks)
    
    
    