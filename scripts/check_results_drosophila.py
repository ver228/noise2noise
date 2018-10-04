#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))


from noise2noise.models import UNet
from noise2noise.trainer import log_dir_root_dflt

import torch
import numpy as np
import cv2
#%%
if __name__ == '__main__':
    model_path = log_dir_root_dflt / 'drosophila_eggs_l1smooth_20181003_004605_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    model_path = log_dir_root_dflt / 'drosophila-eggs_l1smooth_20181003_134919_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    scale_log = (0, 16)
    
    
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #fname = '/Volumes/rescomp1/data/denoising_data/drosophila_eggs/train/1b/1b_t44.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/drosophila_eggs/train/6a_z002/6a_t035_z002.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/drosophila_eggs/train/3b/3b_t110.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/drosophila_eggs/test/1a/1a_t01.tif'
    fname = '/Volumes/rescomp1/data/denoising_data/drosophila_eggs/test/1a/1a_t01.tif'
    
    #%%
    img = cv2.imread(fname, -1)
    
    x = img.astype(np.float32)
    
    x = np.log(x+1)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None, None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    
    #%%
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(x)#, vmin=0, vmax=1)
    axs[1].imshow(xhat)#, vmin=0, vmax=1)
    axs[2].imshow((xhat - x))
    #%%
    
    xhat_l = np.exp(xhat*(scale_log[1] - scale_log[0]) + scale_log[0])
    x_l = np.exp(x*(scale_log[1] - scale_log[0]) + scale_log[0])
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    axs[0].imshow(x_l)
    #xhat[xhat<0.301] = 0.301
    axs[1].imshow(xhat_l)
    axs[2].imshow(xhat_l - x_l)
    