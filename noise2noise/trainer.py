#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path 

from .flow import BasicFlow, SyntheticFluoFlow, InkedFlow, FromTableFlow, MNISTFashionFlow
from .models import UNet, L0AnnelingLoss, BootstrapedPixL2

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
    elif loss_type == 'l0anneling':
        criterion = L0AnnelingLoss(anneling_rate=1/50)
    elif loss_type == 'bootpixl2':
        criterion = BootstrapedPixL2(bootstrap_factor=4)
    
    else:
        raise ValueError(loss_type)
    return criterion

def get_model(model_name):
    if model_name == 'unet':
        model = UNet(n_channels = 1, n_classes = 1)
    elif model_name == 'unet-ch3':
        model = UNet(n_channels = 3, n_classes = 3)
    elif model_name == 'unet-ch4':
        model = UNet(n_channels = 4, n_classes = 4)
    else:
        raise ValueError(model_name)
    return model

def get_flow(data_type, src_root_dir = None):
    def _get_dir(_src_dir):
        if src_root_dir is None:
            return _src_dir
        else:
            return src_root_dir
    print(data_type)
    if data_type == 'drosophila-eggs':
        src_dir = Path.home() / 'workspace/denoising_data/drosophila_eggs/train'
        gen = BasicFlow(_get_dir(src_dir), is_log_transform = True, scale_int = (0, 16), cropping_size=128)
        
    elif data_type == 'drosophila-eggs-N':
        src_dir = Path.home() / 'workspace/denoising_data/drosophila_eggs/train'
        gen = BasicFlow(_get_dir(src_dir), is_log_transform = False, scale_int = (0, 2**16-1), cropping_size=128)
        
    elif data_type == 'worms':
        src_dir = Path.home() / 'workspace/denoising_data/c_elegans/train'
        gen = BasicFlow(_get_dir(src_dir), is_log_transform = False, scale_int = (0, 255))

    elif data_type == 'bertie_worms':
        src_dir = Path.home() / 'workspace/denoising_data/bertie_c_elegans/'
        gen = FromTableFlow(_get_dir(src_dir), is_log_transform = False, scale_int = (0, 255))
        
        
    elif data_type == 'microglia_synthetic':
        src_dir = Path.home() / 'workspace/denoising_data/microglia/syntetic_data'
        gen = SyntheticFluoFlow(_get_dir(src_dir))
    
    elif data_type == 'inked-slides-cmy':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_clipped=False, is_return_rgb=False)
        
    elif data_type == 'inked-slides-cmy-randclip':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_random_clipped=True, is_return_rgb=False)
        
    elif data_type == 'inked_slides':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir))
    elif data_type == 'inked-slides-clipped':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_clipped=True)
    elif data_type == 'inked2real':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(
                    is_clean_output = True, is_symetric_ink=False, 
                    is_clipped=True, is_return_rgb=True)
    
    elif data_type == 'tiny-inked-clipped':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_clipped=True,  is_tiny=True, is_preload=True)
    elif data_type == 'tiny-inked':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_clipped=False,  is_tiny=True, is_preload=True)
    elif data_type == 'tiny-inked-cmy':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_clipped=False, is_return_rgb=False, is_tiny=True, is_preload=True)
    elif data_type == 'tiny-inked-cmy-randclip':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        gen = InkedFlow(_get_dir(src_dir), is_random_clipped=True, is_return_rgb=False, is_tiny=True, is_preload=True)
    
    elif data_type == 'tiny-inked2real':
        src_dir = Path.home() / 'workspace/denoising_data/inked_slides'
        
        gen = InkedFlow(is_tiny=True, is_preload=True, 
                    is_clean_output = True, is_symetric_ink=False, 
                    is_clipped=True, is_return_rgb=True)
    elif data_type == 'mnist-fg-fix':
        gen = MNISTFashionFlow(is_fix_bg = False)
    elif data_type == 'mnist-fg-fix-v1':
         gen = MNISTFashionFlow(
                 is_fix_bg = False,
                  output_size = 256,
                 bg_n_range = (5, 15),
                 int_range = (0.5, 1.0),
                 max_rotation = 90,
                 is_v_flip = True
                 )
    
    elif data_type == 'mnist-fg-fix-v2':
        gen = MNISTFashionFlow(is_fix_bg = False,
                               output_size = 256,
                             bg_n_range = (5, 25),
                             int_range = (1., 1.),
                             max_rotation = 45,
                             is_v_flip = False
                             )
        
    elif data_type == 'mnist-bg-fix':
        gen = MNISTFashionFlow(is_fix_bg = True)

    else:
        raise ValueError(data_type)
    
    print(src_root_dir)
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
        data_src_dir = None,
        init_model_path = None
        ):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    gen = get_flow(data_type, data_src_dir)
    loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    model = get_model(model_name)
    model = model.to(device)
    
    criterion = get_loss(loss_type)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
    bn = '{}_{}_{}_{}_lr{}_wd{}_batch{}'.format(data_type, loss_type, bn, 'adam', lr, weight_decay, batch_size)

    epoch_init = 0 #useful to keep track in restarted models
    if init_model_path:
        #load weights
        init_model_path = Path(init_model_path)
        if not init_model_path.exists():
            init_model_path = log_dir_root / init_model_path
        state = torch.load(str(init_model_path), map_location = dev_str)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_init = state['epoch']
        
        bn = 'R_' + bn
        print('{} loaded...'.format(init_model_path))

    log_dir = log_dir_root / bn
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
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
                'epoch': epoch + epoch_init,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))

if __name__ == '__main__':
    import fire
    fire.Fire(train)