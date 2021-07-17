#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 03:55:47 2021

@author: changxin
"""

import torch

def cal_psnr(x, y):
    '''
    Parameters
    ----------
    x, y are two tensors has the same shape (1, C, H, W)

    Returns
    -------
    score : PSNR.
    '''
    
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse)
    return score