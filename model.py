#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:42:45 2021

@author: changxin
"""

import torch
import math
from torch import nn
from torch.quantization import QuantStub, DeQuantStub

class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()
    
    def forward(self, x):
        return x.clamp(min=0., max=1.)
    
class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Gblock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(Gblock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x
    
class XLSR(nn.Module):
    def __init__(self, SR_rate):
        super(XLSR, self).__init__()

        self.conv0_0 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_1 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_2 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_3 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = ConvRelu(in_channels=48, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3*SR_rate**2, kernel_size=3, padding=1)
        
        self.Gblocks = nn.Sequential(Gblock(32, 32, 4), Gblock(32, 32, 4), Gblock(32, 32, 4))
        self.depth2spcae = nn.PixelShuffle(SR_rate)
        self.clippedReLU = ClippedReLU()
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2/fan_out*0.1)
                torch.nn.init.normal_(m.weight.data, mean=0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
                
    def forward(self, x):
        
        res_conv0_0 = self.conv0_0(x)
        res_conv0_1 = self.conv0_1(x)
        res_conv0_2 = self.conv0_2(x)
        res_conv0_3 = self.conv0_3(x)
        res = torch.cat((res_conv0_0, res_conv0_1, res_conv0_2, res_conv0_3), dim=1)
        
        res = self.conv2(res)
        res = self.Gblocks(res)
        
        res_conv1 = self.conv1(x)
        res = torch.cat((res, res_conv1), dim=1)
        
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.clippedReLU(res)
        
        res = self.depth2spcae(res)
              
        return res


class XLSR_quantization(nn.Module):
    def __init__(self, SR_rate):
        super(XLSR_quantization, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        
        self.conv0_0 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_1 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_2 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_3 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = ConvRelu(in_channels=48, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3*SR_rate**2, kernel_size=3, padding=1)
        
        self.Gblocks = nn.Sequential(Gblock(32, 32, 4), Gblock(32, 32, 4), Gblock(32, 32, 4))
        self.depth2spcae = nn.PixelShuffle(SR_rate)
        self.clippedReLU = ClippedReLU()
        self.cat1 = nn.quantized.FloatFunctional()
        self.cat2 = nn.quantized.FloatFunctional()
        
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
                
    def forward(self, x):
        
        x = self.quant(x)
        res_conv0_0 = self.conv0_0(x)
        res_conv0_1 = self.conv0_1(x)
        res_conv0_2 = self.conv0_2(x)
        res_conv0_3 = self.conv0_3(x)
        
        res = self.cat1.cat((res_conv0_0, res_conv0_1, res_conv0_2, res_conv0_3), dim=1)
        
        res = self.conv2(res)
        res = self.Gblocks(res)
        
        res_conv1 = self.conv1(x)
        res = self.cat2.cat((res, res_conv1), dim=1)
        
        res = self.conv3(res)
        res = self.conv4(res)
        
        res = self.depth2spcae(res)
        res = self.clippedReLU(res)
        
        res = self.dequant(res)
        return res
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvRelu:
                # print("fuse conv and relu in ConvRelu")
                torch.quantization.fuse_modules(m, ['conv', 'relu'], inplace=True)          
            if type(m) == Gblock:
                # print("fuse conv and relu in Gblock")
                torch.quantization.fuse_modules(m, ['conv0', 'relu'], inplace=True)   
                
if __name__ == '__main__':
    device = 'cuda:0'
    model = XLSR(3).to(device)
    # print(model)
    model.eval()
    xx = torch.randn((16,3,32,32)).to(device)
    pred = model(xx)
    print(f"the pred shape is {pred.shape}")
    
