#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:25:05 2021

@author: changxin
"""

import os
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt

def save_res(preds, img_names, save_dir):
    '''

    Parameters
    ----------
    preds : List
        each pred has a shape of 1x3xHxW. BGR
    img_names : List

    Returns
    -------
    None.

    '''
    for pred, img_name in zip(preds, img_names):
        pred_img = pred[0].cpu().numpy().transpose(1,2,0)
        cv2.imwrite(os.path.join(save_dir, img_name), np.uint8(pred_img*255))
    return

def visualize_training(save_dir):
    txt_res = os.path.join(save_dir, 'results.txt')
    with open(txt_res, 'r') as f:
        info = f.readlines()
    epoch, lr, train_loss, valid_loss, psnr = [], [], [], [], []
    
    for line in info[:-1]:
        line = line.strip().split('|')
        epoch.append(int(line[0].split(':')[1]))
        lr.append(float(line[1].split(':')[1]))
        train_loss.append(float(line[2].split(':')[1]))
        valid_loss.append(float(line[3].split(':')[1]))
        psnr.append(float(line[4].split(':')[1]))
    
    fig = plt.figure(figsize=(16, 8), dpi=400)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text('Training loss')
    ax2.title.set_text('Validation loss')
    ax3.title.set_text('PSNR (validation)')
    ax4.title.set_text('learning rate')
    ax1.plot(epoch, train_loss)
    ax2.plot(epoch, valid_loss)
    ax3.plot(epoch, psnr)
    ax4.plot(epoch, lr)
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=200)
    return 