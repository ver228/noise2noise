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
from noise2noise.flow import rgb_cmy, cmy_rgb

import torch
import numpy as np
import cv2
#%%
if __name__ == '__main__':
    n_ch  = 1
     
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181003_232406_unet-3ch_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_093656_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_100258_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_205913_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_212303_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181004_214553_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #model_path = log_dir_root_dflt / 'inked_slides_l2_20181004_220027_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1_20181004_220232_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    #only pos switch color channels + random intensities
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181005_113342_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l0anneling_20181005_113940_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #only pos and negative switch color channels + random intensities
    #model_path = log_dir_root_dflt / 'inked_slides_l1_20181005_115745_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l0anneling_20181005_120410_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides_l1smooth_20181005_132332_unet-ch4_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #no symetric color augmented rgb
    #model_path = log_dir_root_dflt / 'inked_slides_l1_20181005_151524_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #symetric color augmented rgb
    #model_path = log_dir_root_dflt / 'inked_slides_l1_20181005_153004_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    model_path = log_dir_root_dflt / 'inked_slides_l1_20181005_154601_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    scale_log = (0, 255)
    n_ch = 3
    
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #%%
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/Tile003873.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/Tile003991.jpg'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/fake.jpg'
    
    x = cv2.imread(fname, -1)[..., ::-1]/255
    
    #x = rgb_cmy(x)
    #%%
    x = x.astype(np.float32)
    x = np.rollaxis(x, 2, start=0)
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    
#%%
    xhat_r = np.rollaxis(xhat, 0, start=3)
    xhat_r = np.clip(xhat_r, 0, 1)
    #xhat_r = cmy_rgb(xhat_r)
    
    xr = np.rollaxis(x, 0, start=3)
    #xr = cmy_rgb(xr)
    
    
    real_fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/clean/Tile001421.jpg'
    img_real = cv2.imread(real_fname, -1)[..., ::-1].astype(np.float32)/255
    #%%
#    with torch.no_grad():
#        Xbase = model(torch.zeros(X.shape))
#    
#    xbase = Xbase.squeeze().detach().numpy()
#    
#    rr = xhat-xbase
#    
#    rr = np.rollaxis(rr, 0, start=3)
#    rr = np.clip(rr, 0, 1)
#    
#    rr = cmy_rgb(rr)
#    #plt.imshow(rr)
#    xhat_r = rr
#    
    #%%
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(xr)#, vmin=0, vmax=1)
    axs[0].set_title('input')
    axs[1].imshow(xhat_r)#, vmin=0, vmax=1)
    axs[1].set_title('output')
    axs[2].imshow(img_real)
    axs[2].set_title('ground truth')
    