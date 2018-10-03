#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""

from pathlib import Path 

from .flow import BasicFlow
from .models import UNet

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import datetime
import shutil
import tqdm

log_dir_root_dflt = Path.home() / 'workspace/denoising_data/results/'

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)
        
def get_loss(loss_type):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(loss_type)
    
    return criterion

def get_model(model_name):
    if model_name == 'unet':
        model = UNet(n_channels = 1, n_classes = 1)
    else:
        raise ValueError(model_name)
    return model

def get_flow(data_type):
    if data_type == 'drosophila_eggs':
        src_root_dir = Path.home() / 'workspace/denoising_data/drosophila_eggs/train'
        gen = BasicFlow(src_root_dir, is_log_transform = True, scale_int = (0, 16), cropping_size=128)
    if data_type == 'drosophila-eggs-N':
        src_root_dir = Path.home() / 'workspace/denoising_data/drosophila_eggs/train'
        gen = BasicFlow(src_root_dir, is_log_transform = False, scale_int = (0, 2**16-1), cropping_size=128)
    elif data_type == 'worms':
        src_root_dir = Path.home() / 'workspace/denoising_data/c_elegans/train'
        gen = BasicFlow(src_root_dir, is_log_transform = False, scale_int = (0, 255))
    else:
        raise ValueError(data_type)
    return gen

def train(
        data_type = 'drosophila_eggs',
        loss_type = 'l1smooth',
        log_dir_root = log_dir_root_dflt,
        cuda_id = 0,
        batch_size = 16,
        model_name = 'unet',
        lr = 1e-4, 
        weight_decay = 0.0,
        n_epochs = 2000,
        num_workers = 1,
        is_to_align = False,
        ):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    gen = get_flow(data_type)
    loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    model = get_model(model_name)
    model = model.to(device)
    
    criterion = get_loss(loss_type)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
    bn = '{}_{}_{}_{}_lr{}_wd{}_batch{}'.format(data_type, loss_type, bn, 'adam', lr, weight_decay, batch_size)
    log_dir = log_dir_root / bn
    logger = SummaryWriter(log_dir = str(log_dir))
    
    #%%
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        pbar = tqdm.tqdm(loader)
        
        avg_loss = 0
        frac_correct = 0
        for X, target in pbar:
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            avg_loss += loss.item()
            
            
        
        avg_loss /= len(loader)
        frac_correct /= len(gen)
        tb = [('train_epoch_loss', avg_loss)]
        
        for tt, val in tb:
            logger.add_scalar(tt, val, epoch)
        
        
        
        desc = 'epoch {} , loss={}, acc={}'.format(epoch, avg_loss, frac_correct)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))

if __name__ == '__main__':
    import fire
    fire.Fire(train)