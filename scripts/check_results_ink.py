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
from noise2noise.flow import rgb_cmyk, cmyk_rgb

import torch
import numpy as np
import cv2
#%%
if __name__ == '__main__':
    n_ch  = 1
     
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181003_232406_unet-3ch_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_093656_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_100258_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    
    scale_log = (0, 255)
    n_ch = 4
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/Tile003873.jpg'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/fake.jpg'
    
    img = cv2.imread(fname, -1)[..., ::-1]
    
    x = rgb_cmyk(img)
    
    x = x.astype(np.float32)
    x = np.rollaxis(x, 2, start=0)
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    

    xhat_r = np.rollaxis(xhat, 0, start=3)
    xhat_r = cmyk_rgb(xhat_r)
    
    xr = np.rollaxis(x, 0, start=3)
    xr = cmyk_rgb(xr)
    #%%
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(xr)#, vmin=0, vmax=1)
    axs[1].imshow(xhat_r)#, vmin=0, vmax=1)
    axs[2].imshow((xhat_r - xr))
    
    