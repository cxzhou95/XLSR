#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 21:28:08 2021

@author: changxin
"""

import os
import argparse
import torch
import time

from model import XLSR
from dataset import create_dataloader
from metric import cal_psnr
from visualization import save_res

  
def test(model, dataloader, device, txt_path):
    pred_list = []
    name_list = []
    avg_psnr = 0.
    avg_time = 0.
    with torch.no_grad():   
        print("warm up ...")
        random_input = torch.randn(1, 3, 640, 360).to(device)
        # warm up
        for _ in range(10):
            model(random_input)
            
        with torch.autograd.profiler.profile() as prof:
            model(random_input)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        
        print("Start testing the model speed on 640*360 input ...")
        test_t = 0.
        for idx in range(100):
            if device != 'cpu':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(random_input)
            if device != 'cpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"Inference #{idx}, inference time: {1000*(t1-t0):.2f}ms")
            test_t += t1 - t0
        print(f"Average inference time on 640*360 input: {1000*test_t/100:.2f}ms") 
        with open(txt_path, 'a') as f:
            f.write(f"Average inference time on 640*360 input: {1000*test_t/100:.2f}ms" + '\n')
                
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
    parser.add_argument('--save-dir', type=str, default='exp/OneCyclicLR_exp0', help='hyperparameters path')
    parser.add_argument('--SR-rate', type=int, default=3, help='the scale rate for SR')
    parser.add_argument('--model', type=str, default='', help='the path to the saved model')
    parser.add_argument('--device', type=str, default='cpu', help='gpu id or "cpu"')
    opt = parser.parse_args()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    
    # cuDnn configurations
    if opt.device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    # txt file to record training process
    txt_path = os.path.join(opt.save_dir, 'test_res.txt')
    if os.path.exists(txt_path):
        os.remove(txt_path)
    
    # folder to save the predicted HR image in the validation
    test_folder = os.path.join(opt.save_dir, 'test_res')
    os.makedirs(test_folder, exist_ok=True)
       
    device = 'cuda:' + str(opt.device) if opt.device != 'cpu' else 'cpu'
    model = XLSR(opt.SR_rate)
    
    # load pretrained model
    if opt.model.endswith('.pt') and os.path.exists(opt.model):
        model.load_state_dict(torch.load(opt.model, map_location=device))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.save_dir, 'best.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    test_dataloader = create_dataloader('test', opt.SR_rate, False, batch_size=1, shuffle=False, num_workers=1)
           
    # evaluate
    pred_list, name_list = test(model, test_dataloader, device, txt_path)
                 
    print("Saving the predicted HR images")
    save_res(pred_list, name_list, test_folder)
    print(f"Testing is done!, predicted HR images are saved in {test_folder}")
    
        

