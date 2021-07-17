#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:18:20 2021

@author: changxin
"""

import os
import yaml
import argparse
import torch
import time
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model import XLSR
from dataset import create_dataloader
from loss import CharbonnierLoss
from metric import cal_psnr
from visualization import save_res, visualize_training

def train(model, dataloader, criteria, device, optimizer, scheduler):
    loss_epoch = 0.
    for LR_img, HR_img, _ in dataloader:

        optimizer.zero_grad()
        LR_img, HR_img = LR_img.to(device).float(), HR_img.to(device).float()
        HR_pred = model(LR_img)
        loss = criteria(HR_pred, HR_img)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_epoch += loss.item()
    loss_epoch /= len(dataloader)
    lr_epoch = scheduler.get_last_lr()[0]
    # scheduler.step()
    return loss_epoch, lr_epoch

def validation(model, dataloader, criteria, device):
    loss_epoch = 0.
    psnr_epoch = 0.
    pred_list = []
    name_list = []
    with torch.no_grad():
        for LR_img, HR_img, img_name in dataloader:
            LR_img, HR_img = LR_img.to(device).float(), HR_img.to(device).float()
            HR_pred = model(LR_img)
            loss = criteria(HR_pred, HR_img)
            loss_epoch += loss.item()
            psnr_epoch += cal_psnr(HR_pred, HR_img).item()
            pred_list.append(HR_pred)
            name_list += img_name
    loss_epoch /= len(dataloader)
    psnr_epoch /= len(dataloader)
    return loss_epoch, psnr_epoch, pred_list, name_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/first-try', help='the dir where the model and ')
    parser.add_argument('--SR-rate', type=int, default=3, help='the scale rate for SR')
    parser.add_argument('--pretrained-model', type=str, default='', help='the path to the pretrained model')
    parser.add_argument('--epochs', type=int, default=5000, help='the number of total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--augment', action='store_true', help='whether to use augmentation such as random crop, intensity change, ...')
    parser.add_argument('--workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--device', type=int, default='0', help='gpu id')
    parser.add_argument('--div-factor', type=float, default=50, help='div_factor, initial lr = lr_max / div_factor')
    parser.add_argument('--final-div-factor', type=float, default=0.5, help='the final learning rate = initial lr / final_div_factor')
    parser.add_argument('--lr-max', type=float, default=25e-04, help='the maximum learning rate for OneCycleLR')
    parser.add_argument('--pct-epoch', type=int, default=50, help='the epoch that has the the maximum learning rate, before this epoch, lr increases, after this epoch, lr decreases')
    opt = parser.parse_args()
    
    if os.path.exists(opt.save_dir):
        print(f"Warning: {opt.save_dir} exists, please delete it manually if it is useless.")

    os.makedirs(opt.save_dir, exist_ok=False)
    
    # save hyp-parameter
    with open(os.path.join(opt.save_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(opt, f, sort_keys=False)
    
    # txt file to record training process
    txt_path = os.path.join(opt.save_dir, 'results.txt')
    if os.path.exists(txt_path):
        os.remove(txt_path)
    
    # folder to save the predicted HR image in the validation
    valid_folder = os.path.join(opt.save_dir, 'valid_res')
    os.makedirs(valid_folder, exist_ok=True)
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)  # set environment variable
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % str(opt.device)  # check availablity
    device = 'cuda'
    model = XLSR(opt.SR_rate)
    
    # load pretrained model
    if opt.pretrained_model.endswith('.pt') and os.path.exists(opt.pretrained_model):
        # filter conv4 weights which have different conv channels
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrained_model)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pretrained model {opt.pretrained_model}" )
        
    model.to(device)
    
    train_dataloader = create_dataloader('train', opt.SR_rate, opt.augment, opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid_dataloader = create_dataloader('valid', opt.SR_rate, False, 1, shuffle=False, num_workers=1)
    
    criteria = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_max/opt.div_factor, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.OneCycleLR(optimizer, opt.lr_max, epochs=opt.epochs, steps_per_epoch=len(train_dataloader), pct_start=opt.pct_epoch/opt.epochs, anneal_strategy='cos', \
                                        cycle_momentum=False, div_factor=opt.div_factor, final_div_factor=opt.final_div_factor)
    
    best_psnr = 0.
    for idx in range(1, opt.epochs+1):
        t0 = time.time()
        train_loss_epoch, lr_epoch = train(model, train_dataloader, criteria, device, optimizer, scheduler)
        t1 = time.time()
        valid_loss_epoch, psnr_epoch, pred_HR, img_names = validation(model, valid_dataloader, criteria, device)
        t2 = time.time()
        print(f"Epoch: {idx} | lr: {lr_epoch:.5f} | training loss: {train_loss_epoch:.5f} | validation loss: {valid_loss_epoch:.5f} | PSNR: {psnr_epoch:.3f} | Time: {t2-t0:.1f}")
        with open(txt_path, 'a') as f:
            f.write(f"Epoch: {idx} | lr: {lr_epoch:.5f} | training loss: {train_loss_epoch:.5f} | validation loss: {valid_loss_epoch:.5f} | PSNR: {psnr_epoch:.3f} | Time: {t2-t0:.1f}" +'\n')
        
        if psnr_epoch > best_psnr:
            best_psnr = psnr_epoch
            # save model
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'best.pt'))
            # save predicted HR image on validation set
            save_res(pred_HR, img_names, valid_folder)
        del pred_HR
    
    # visualize the training process
    visualize_training(opt.save_dir)
    print(f"Training is finished, the best PSNR is {best_psnr:.3f}")
    with open(txt_path, 'a') as f:
            f.write(f"Training is finished, the best PSNR is {best_psnr:.3f}")
        
