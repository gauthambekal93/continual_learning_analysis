# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:34:28 2026

@author: gauthambekal93
"""
import torch
from torch import nn
from models.activation import CReLU, CustomTanh
from models.resnet import ResBlock
import numpy as np

class MLP(nn.Module):

    def __init__(
        self,
        in_dim=2,
        width=10,
        depth=4,
        activation="relu",
        residual=False,
        norm="none",
        dropout=0.0
    ):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU

        elif activation == "leaky_relu":
            act = lambda: nn.LeakyReLU(0.01)
        
        elif activation == "elu":
            act = lambda: nn.ELU(alpha=1.0)
        
        elif activation == "prelu":
            act = lambda: nn.PReLU(num_parameters=1, init=0.25)
        
        elif activation == "tanh":
            act = nn.Tanh
        
        elif activation == "custom_tanh":
            act = CustomTanh
            
        elif activation == "softsign":
            act = nn.Softsign
            
        elif activation == "silu":
            act = nn.SiLU
    
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
        
        def make_dropout():
            if dropout > 0:
                return nn.Dropout(p=dropout)
            else:
                return nn.Identity()
            
        layers = []

        if depth == 1:
            layers.append(nn.Linear(in_dim, 2))

        else:

            if activation != "concat_relu":
                layers.append(nn.Linear(in_dim, width))
                layers.append(make_norm(width))
                layers.append(act())
                layers.append(make_dropout())

            else:
                layers.append(nn.Linear(in_dim, width // 2))
                layers.append(make_norm(width // 2))
                layers.append(act())
                layers.append(make_dropout())

            for _ in range(depth - 2):

                if residual and activation != "concat_relu":

                    layers.append(
                        ResBlock(
                            width,
                            activation=activation,
                            norm=norm
                        )
                    )

                elif residual is False and activation != "concat_relu":

                    layers.append(nn.Linear(width, width))
                    layers.append(make_norm(width))
                    layers.append(act())
                    layers.append(make_dropout())

                elif residual is False and activation == "concat_relu":

                    layers.append(nn.Linear(width, width // 2))
                    layers.append(make_norm(width // 2))
                    layers.append(act())
                    layers.append(make_dropout())

                else:
                    raise ValueError(
                        "Residual with Concat ReLU not implemented."
                    )

            layers.append(nn.Linear(width, 2))

        self.net = nn.Sequential(*layers)

    def forward( self, x, return_pre_act = False, return_layer_out = False, device="cpu"):
        result = {}
        #pre_activations = {}
        dead_neuron_frac = {}
        mean_act = {}
        pos_neuron_frac = {}
        prelu_param = {}
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
    
        for layer_num, layer in enumerate(self.net):
           
            layer_name = layer.__class__.__name__
            
            x = layer(x)

            if return_layer_out and layer_name.lower()!='identity':
                x = x.detach()
                dead_neuron_frac[str(layer_num) + "_" + layer_name + "_dead_frac"] =  (x == 0).all(dim = 0).float().mean().item()
                pos_neuron_frac[str(layer_num) + "_" + layer_name + "pos_frac"]  = (x> 0).float().mean().item()
                mean_act[str(layer_num) + "_" + layer_name + "abs_mean_act"] = x.abs().mean().item() #x.abs().mean().item()
                
                if layer_name.lower() == 'prelu':
                   prelu_param[str(layer_num) + "_" + layer_name + "_param"] = layer.weight.detach().item()
                   
                
            #if return_pre_act and isinstance(layer, ( nn.ReLU, nn.LeakyReLU, CReLU ) ):
            #    pre_activations[str(layer_num) + "_" + layer_name] = x.detach()
        
        if return_layer_out:
            result.update({"mean_act" : mean_act})
            result.update({"dead_neuron_frac" : dead_neuron_frac})
            result.update({"pos_neuron_frac" : pos_neuron_frac})
            result.update({"prelu_param" : prelu_param})
        #if return_pre_act:
        #    result.update({"pre_activations":pre_activations})
        
        result.update({"pred":x})

        return result