#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:19:08 2021

@author: changxin
"""

import torch
from torch import nn

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=0.01):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred, gt):
 
        loss = torch.sqrt((pred - gt)**2 + self.eps).mean()
        
        return loss.mean()