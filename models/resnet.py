# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:59:35 2026

@author: gauthambekal93
"""

from torch import nn

class ResBlock(nn.Module):
    def __init__(self, width, activation="relu", norm="none"):
        super().__init__()

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            self.act = nn.Tanh()
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