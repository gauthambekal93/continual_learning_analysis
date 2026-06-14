# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:12:07 2026

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:26:21 2026

@author: gauthambekal93
"""



import random
import numpy as np
import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm
import torch.nn.functional as F


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    
# ---------------- seeds ----------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------- task generator ----------------
def sample_task_params():
    w1 = np.random.randn(2).astype(np.float32)
    w1 /= np.linalg.norm(w1) + 1e-9

    w2 = np.random.randn(2).astype(np.float32)
    w2 /= np.linalg.norm(w2) + 1e-9

    b1 = np.random.uniform(0, 2 * np.pi)
    b2 = np.random.uniform(0, 2 * np.pi)

    return w1, w2, b1, b2


def sample_from_task(params, n=512, noise=0.02):
    w1, w2, b1, b2 = params

    X = np.random.randn(n, 2).astype(np.float32)

    score = (
        np.sin(3.0 * (X @ w1) + b1)
        + 0.5 * np.cos(5.0 * (X @ w2) + b2)
    )

    y = (score > 0).astype(np.int64)

    if noise > 0:
        flip = np.random.rand(n) < noise
        y[flip] = 1 - y[flip]

    return X, y


class ResBlock(nn.Module):
    def __init__(self, width, activation="relu", norm="none"):
        super().__init__()

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.01)
        else:
            raise ValueError("Residual block only supports relu/leaky_relu for now")

        self.lin = nn.Linear(width, width)

        if norm == "layernorm":
            self.norm = nn.LayerNorm(width)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError("unknown norm")

    def forward(self, x):
        z = self.lin(x)
        z = self.norm(z)
        z = self.act(z)
        return x + z
    
    
class MLP(nn.Module):
    def __init__(
        self,
        in_dim=2,
        width=10,
        depth=4,
        activation="relu",
        residual=False,
        norm="none"
    ):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU
        elif activation == "leaky_relu":
            act = lambda: nn.LeakyReLU(0.01)
        elif activation == "concat_relu":
            act = CReLU
        else:
            raise ValueError("unknown activation")

        def make_norm(dim):
            if norm == "layernorm":
                return nn.LayerNorm(dim)
            elif norm == "none":
                return nn.Identity()
            else:
                raise ValueError("unknown norm")

        layers = []

        if depth == 1:
            layers.append(nn.Linear(in_dim, 2))

        else:
            layers.append(nn.Linear(in_dim, width))
            layers.append(make_norm(width))
            layers.append(act())

            for _ in range(depth - 2):
                if residual and activation != "concat_relu":
                    layers.append(
                        ResBlock(
                            width,
                            activation=activation,
                            norm=norm
                        )
                    )
                else:
                    layers.append(nn.Linear(width, width))
                    layers.append(make_norm(width))
                    layers.append(act())

            layers.append(nn.Linear(width, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    

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


# ---------------- training ----------------
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


# ---------------- checkpoint ----------------
def save_checkpoint(net, opt, rows, out_dir, seed, epoch, depth, width, activation, norm, wd, task):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        out_dir,
        f"results_seed{seed}_resnet_epoch{epoch}_depth{depth}_width{width}_activation_{activation}_norm_{norm}_wd_{wd}.csv"
    )
    df.to_csv(csv_path, index=False)

    latest_path = os.path.join(
        out_dir,
        f"latest_seed{seed}_resnet_epoch{epoch}_depth{depth}_width{width}_activation_{activation}_norm_{norm}__wd_{wd}.pt"
    )

    torch.save({
        "task": task,
        "seed": seed,
        "depth": depth,
        "width": width,
        "activation": activation,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "results": rows,
    }, latest_path)

    tqdm.write(f"Saved checkpoint at task {task}")


# ---------------- experiment ----------------
def run_experiment(
    seed=20,
    depth=4,
    width=10,
    num_tasks=100000,
    epochs_per_task=5,
    batch_size=64,
    n_train=512,
    n_test=2048,
    noise=0.02,
    lr=1e-3,
    wd=0.0,
    activation="relu",
    device="cpu",
    out_dir="checkpoints_random_nonlinear_task",
    save_every=1000,
    print_every=10,
    residual = False,
    norm="layernorm"
):
    set_seed(seed)

    net = MLP(width=width, depth=depth, activation=activation, residual = residual, norm = norm).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    rows = []

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
            "task": t,
            "depth": depth,
            "width": width,
            "seed": seed,
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
            save_checkpoint(net, opt, rows, out_dir, seed, epochs_per_task, depth, width, activation, norm, wd, t + 1)

    save_checkpoint(net, opt, rows, out_dir, seed, epochs_per_task, depth, width, activation, norm, wd, num_tasks)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_dfs = []

    for depth in [20]:
        df = run_experiment(
        seed=20,
        depth=depth,
        width=10,
        num_tasks=100000,
        epochs_per_task=20,
        batch_size=64,
        n_train=512,
        n_test=2048,
        noise=0.0,
        lr=1e-2,
        wd=0.0,
        activation="relu",
        device="cpu",
        out_dir="checkpoint_resnet_normalization",
        save_every=100,
        residual=True,
        norm="none"
    )
        all_dfs.append(df)
    