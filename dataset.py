#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:24:28 2021

@author: changxin
"""

import cv2
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def create_dataloader(split, SR_rate, augment, batch_size=1, shuffle=False, num_workers=1, pin_memory=True):
    dataset = DIV2K(split, SR_rate, augment)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader
    
def random_crop(LR_img, HR_img, crop_size, SR_rate):
    # check the shape
    LR_h, LR_w = LR_img.shape[:2]
    HR_h, HR_w = HR_img.shape[:2]
    assert LR_h * SR_rate == HR_h and LR_w * SR_rate == HR_w, 'SR_rate is not correct for LR and HR image'
    # check the crop size
    new_LR_h, new_LR_w = crop_size
    assert new_LR_h <= LR_h and new_LR_w <= LR_w, 'crop_size is too large'
    
    y1 = random.randint(0, LR_h - new_LR_h)
    x1 = random.randint(0, LR_w - new_LR_w)
    
    LR_crop = LR_img[y1:y1 + new_LR_h, x1:x1 + new_LR_w, :]
    HR_crop = HR_img[SR_rate * y1:SR_rate * (y1 + new_LR_h), SR_rate * x1:SR_rate * (x1 + new_LR_w), :]
    
    return LR_crop, HR_crop
    
class DIV2K(Dataset):  # for training/testing
    def __init__(self, split, SR_rate, augment=False):
        
        self.split = split
        self.SR_rate = SR_rate
        self.augment = augment
        self.intensity_list = [1.0, 0.7, 0.5]
        self.crop_size = [32, 32]
        
        # data split
        if split == 'train':
            self.LR_dir = os.path.join('./DIV2K/DIV2K_train_LR_bicubic', 'X'+str(SR_rate))
            self.HR_dir = './DIV2K/DIV2K_train_HR'
            self.img_names = sorted(os.listdir(self.HR_dir))[:792]
        elif split == 'valid':
            self.LR_dir = os.path.join('./DIV2K/DIV2K_train_LR_bicubic', 'X'+str(SR_rate))
            self.HR_dir = './DIV2K/DIV2K_train_HR'
            self.img_names = sorted(os.listdir(self.HR_dir))[792:]
        elif split == 'test':
            self.LR_dir = os.path.join('./DIV2K/DIV2K_valid_LR_bicubic', 'X'+str(SR_rate))
            self.HR_dir = './DIV2K/DIV2K_valid_HR'
            self.img_names = sorted(os.listdir(self.HR_dir))
        else:
            raise NameError('data split must be "train", "valid" or "test". ')

        

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, index):
        
        if self.split == 'train':
            LR_img = cv2.imread(os.path.join(self.LR_dir, self.img_names[index][:-4]+'x'+str(self.SR_rate)+'.png')) / 255.
            HR_img = cv2.imread(os.path.join(self.HR_dir, self.img_names[index])) / 255.
            
            if self.augment:
                # random crop
                LR_img, HR_img = random_crop(LR_img, HR_img, self.crop_size, self.SR_rate) 
            
                # geometric transformations
                if random.random() < 0.5: # hflip
                    LR_img, HR_img = LR_img[:, ::-1, :], HR_img[:, ::-1, :]
                if random.random() < 0.5: # vflip
                    LR_img, HR_img = LR_img[::-1, :, :], HR_img[::-1, :, :]
                if random.random() < 0.5: # rot90
                    LR_img, HR_img = LR_img.transpose(1, 0, 2), HR_img.transpose(1, 0, 2)
            
                # intensity scale
                intensity_scale = random.choice(self.intensity_list)
                LR_img *= intensity_scale
                HR_img *= intensity_scale
            
            
        else:
            LR_img = cv2.imread(os.path.join(self.LR_dir, self.img_names[index][:-4]+'x'+str(self.SR_rate)+'.png')) / 255.
            HR_img = cv2.imread(os.path.join(self.HR_dir, self.img_names[index])) / 255.
        
        # Convert
        LR_img = np.ascontiguousarray(LR_img.transpose(2, 0, 1)) # HWC => CHW
        HR_img = np.ascontiguousarray(HR_img.transpose(2, 0, 1)) 

        return torch.from_numpy(LR_img), torch.from_numpy(HR_img), self.img_names[index]

if __name__ == '__main__':
    os.makedirs('./test_dataloader/', exist_ok=True)
    train_dataloader = create_dataloader('train', 3, False, batch_size=1, shuffle=False, num_workers=1)
    print(f"len(train): {len(train_dataloader)}")
    LR_img, HR_img, img_names = next(iter(train_dataloader))
    print(f"LR_img shape: {LR_img.size()}")
    print(f"HR_img shape: {HR_img.size()}")
    print(img_names)
    LR_img = LR_img[0].numpy().transpose(1, 2, 0)
    HR_img = HR_img[0].numpy().transpose(1, 2, 0)
    cv2.imwrite('./test_dataloader/LR_img.png', np.uint8(LR_img*255))
    cv2.imwrite('./test_dataloader/HR_img.png', np.uint8(HR_img*255))
