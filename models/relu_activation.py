# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:12:20 2026

@author: gauthambekal93
"""

import torch
from torch import nn
import torch.nn.functional as F

class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    