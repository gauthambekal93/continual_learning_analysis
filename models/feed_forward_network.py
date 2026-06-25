# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:00:31 2026

@author: gauthambekal93
"""


from torch import nn
from models.activation import CReLU
from models.resnet import ResBlock

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
            elif norm == "batchnorm":
                return nn.BatchNorm1d(dim)
            elif norm == "none":
                return nn.Identity()
            else:
                raise ValueError("unknown norm")

        layers = []

        if depth == 1:
            layers.append(nn.Linear(in_dim, 2))

        else:
            if activation != 'concat_relu':
                layers.append(nn.Linear(in_dim, width))
                layers.append(make_norm(width))
                layers.append(act())
            else:
                layers.append(nn.Linear(in_dim, width // 2 ))
                layers.append(make_norm(width))
                layers.append(act())
                
                
            for _ in range(depth - 2):
                if residual is True and activation != "concat_relu":
                    layers.append(
                        ResBlock(
                            width,
                            activation=activation,
                            norm=norm
                        )
                    )
                if residual is False and activation != "concat_relu":
                    layers.append(nn.Linear(width, width))
                    layers.append(make_norm(width))
                    layers.append(act())
                
                if residual is False and activation == "concat_relu":
                    layers.append(nn.Linear(width, width // 2))
                    layers.append(make_norm(width))
                    layers.append(act())
                
                if residual is True and activation == "concat_relu":
                    raise ValueError("Residual with ConCat Relu Implemented Yet !")
                    
            layers.append(nn.Linear(width, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    