#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 21:28:08 2021

Quantization-aware training
Reference: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training

@author: changxin
"""

import os
import argparse
import torch
import time

from model import XLSR, XLSR_quantization
from dataset import create_dataloader
from metric import cal_psnr
from torch.profiler import profile, record_function, ProfilerActivity
from visualization import save_res
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from loss import CharbonnierLoss

# # Setup warnings
import warnings


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def train_one_epoch(model, criterion, optimizer, train_dataloader, device='cpu', scheduler=None):

    loss_epoch = 0.
    for LR_img, HR_img, _ in train_dataloader:
        optimizer.zero_grad()
        LR_img, HR_img = LR_img.to(device).float(), HR_img.to(device).float()
        HR_pred = model(LR_img)
        loss = criteria(HR_pred, HR_img)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_epoch += loss.item()
    loss_epoch /= len(train_dataloader)
    # scheduler.step()
    return loss_epoch

def measure_speed(model, device):
    with torch.no_grad():   
        # print("warm up ...")
        random_input = torch.randn(1, 3, 256, 256).to(device)
        # warm up
        for _ in range(5):
            model(random_input)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                random_input = torch.randn(1, 3, 640, 360).to(device)
                model(random_input)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))

def validation(model, dataloader, criteria, device='cpu'):
    loss_epoch = 0.
    psnr_epoch = 0.
    with torch.no_grad():
        for LR_img, HR_img, _ in dataloader:
            LR_img, HR_img = LR_img.to(device).float(), HR_img.to(device).float()
            HR_pred = model(LR_img)
            loss = criteria(HR_pred, HR_img)
            loss_epoch += loss.item()
            psnr_epoch += cal_psnr(HR_pred, HR_img).item()
    loss_epoch /= len(dataloader)
    psnr_epoch /= len(dataloader)
    return loss_epoch, psnr_epoch

def test(model, dataloader, device, txt_path):
    pred_list = []
    name_list = []
    avg_psnr = 0.
    avg_time = 0.
    with torch.no_grad():   
        random_input = torch.randn(1, 3, 640, 360).to(device)
        print("Start testing the model speed on 640*360 input ...")
        test_t = 0.
        for idx in range(10):
            if device != 'cpu':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(random_input)
            if device != 'cpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"Inference #{idx}, inference time: {1000*(t1-t0):.2f}ms")
            test_t += t1 - t0
        print(f"Average inference time on 640*360 input: {1000*test_t/10:.2f}ms") 
        with open(txt_path, 'a') as f:
            f.write(f"Average inference time on 640*360 input: {1000*test_t/10:.2f}ms" + '\n')
                
        print("Start the inference ...")
        for LR_img, HR_img, img_name in dataloader:
            LR_img, HR_img = LR_img.to(device).float(), HR_img.to(device).float()
            if device != 'cpu':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            HR_pred = model(LR_img)
            if device != 'cpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            psnr = cal_psnr(HR_pred, HR_img).item()
            inference_time = t1 - t0
            print(f"PSRN on {img_name}: {psnr:.3f}, inference time: {1000*inference_time:.2f}ms")
            with open(txt_path, 'a') as f:
                f.write(f"PSRN on {img_name}: {psnr:.3f}, inference time: {1000*inference_time:.2f}ms" + '\n')
            avg_psnr += psnr
            avg_time += inference_time
            pred_list.append(HR_pred)
            name_list += img_name
    avg_psnr /= len(test_dataloader)
    avg_time /= len(test_dataloader)
    print(f"Average PSRN: {avg_psnr:.3f}, average inference time: {1000*avg_time:.2f}ms")
    with open(txt_path, 'a') as f:
        f.write(f"Average PSRN: {avg_psnr:.3f}, average inference time: {1000*avg_time:.2f}ms")
    return pred_list, name_list
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/OneCyclicLR', help='hyperparameters path')
    parser.add_argument('--SR-rate', type=int, default=3, help='the scale rate for SR')
    parser.add_argument('--model', type=str, default='', help='the path to the saved model')
    parser.add_argument('--epochs', type=int, default=400, help='')
    opt = parser.parse_args()
    
    torch.set_num_threads(4) 
    
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
        )
    warnings.filterwarnings(
        action='default',
        module=r'torch.quantization'
        )
    
    # txt file to record process
    txt_path = os.path.join(opt.save_dir, 'quantizatgion_res.txt')
    if os.path.exists(txt_path):
        os.remove(txt_path)
    # folder to save the predicted HR image in the validation
    test_folder = os.path.join(opt.save_dir, 'quantizatgion_res')
    os.makedirs(test_folder, exist_ok=True)
    
    train_dataloader = create_dataloader('train', opt.SR_rate, True, 16, shuffle=True, num_workers=0, pin_memory=False)
    valid_dataloader = create_dataloader('valid', opt.SR_rate, False, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_dataloader = create_dataloader('test', opt.SR_rate, False, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    device = 'cpu'
    # Specify random seed for repeatable results
    torch.manual_seed(191009)
    model = XLSR_quantization(opt.SR_rate)
    os.makedirs(opt.save_dir, exist_ok=True)    
    # txt file to record training process
    criteria = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.OneCycleLR(optimizer, 5e-4, epochs=opt.epochs, steps_per_epoch=len(train_dataloader), pct_start=50/opt.epochs, anneal_strategy='cos', \
                                        cycle_momentum=False, div_factor=50, final_div_factor=0.5)
   
    # load pretrained model
    if opt.model.endswith('.pt') and os.path.exists(opt.model):
        model.load_state_dict(torch.load(opt.model, map_location=device))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.save_dir, 'best.pt'), map_location=device))
        
    model.to(device)
    model.fuse_model()
      
    
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    best_psnr = 0.
    for nepoch in range(opt.epochs):
        if nepoch > 50:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        t0 = time.time()
        loss_train = train_one_epoch(model, criteria, optimizer, train_dataloader, device, scheduler)
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        quantized_model.eval()
        loss_valid, psnr = validation(quantized_model, valid_dataloader, criteria)
        t1 = time.time()
        print(f"Epoch: {nepoch} | training loss: {loss_train:.5f} | validation loss: {loss_valid:.5f} | PSNR: {psnr:.3f} | Time: {t1-t0:.1f}")
        if psnr > best_psnr:
            torch.jit.save(torch.jit.script(quantized_model), os.path.join(opt.save_dir, 'quantized_model.pt'))
    
    
    quantized_model = torch.jit.load(os.path.join(opt.save_dir, 'quantized_model.pt'))
    
    quantized_model.eval()
    measure_speed(quantized_model, device)
     
    
    # evaluate
    pred_list, name_list = test(quantized_model, test_dataloader, device, txt_path)
    
         
    print("Saving the predicted HR images")
    save_res(pred_list, name_list, test_folder)
    print(f"Testing is done!, predicted HR images are saved in {test_folder}")

    
        
