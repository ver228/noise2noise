#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""

from torch import nn
import torch
class L0AnnelingLoss(nn.Module):
    def __init__(self, anneling_rate=1/50):
        super().__init__()
        
        self.anneling_rate = anneling_rate
        self._n_calls = 0
        self._init_gamma = 2
        self._last_gamma = 0
        self._eps = 1e-8
    
    def forward(self, input_v, target):
        gamma = max(self._init_gamma - self._n_calls*self.anneling_rate, self._last_gamma)
        self._n_calls += 1
        
        return ((input_v-target).abs() + self._eps).pow(gamma).sum()



class BootstrapedPixL2(nn.Module):
    '''bootstrapped pixel-wise L2 loss'''
    def __init__(self, bootstrap_factor=4):
        super().__init__()
        self.bootstrap_factor = bootstrap_factor
        
    def forward(self, input_v, target):
        mat_l2 = torch.pow(input_v-target,2)
        mat_l2 = mat_l2.view(mat_l2.shape[0],-1)
        out, _ = torch.topk(mat_l2, 4, dim=1)
        return out.sum()