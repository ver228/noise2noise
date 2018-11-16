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
    #tiny symetric color augmented rgb clipped 0,1
    #model_path = log_dir_root_dflt / 'inked_slides/tiny/inked_slides_l1_20181005_154601_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides/tiny/R_inked_slides_l1_20181010_073730_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides/tiny/R_inked-slides-clipped_l1_20181012_111351_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #model_path = log_dir_root_dflt / 'tiny-inked-clipped_l1_20181026_150019_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked-cmy-randclip_l1_20181026_163230_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked-cmy_l1_20181026_163228_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked_l1_20181026_163230_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    #tiny symetric color augmented rgb no clipped (faster flow)
    #model_path = log_dir_root_dflt / 'inked_slides/tiny/inked_slides_l1_20181012_101949_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #full asymetric
    #model_path = log_dir_root_dflt / 'inked_slides/inked-slides-clipped_l1smooth_20181005_173844_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides/inked-slides-clipped_l1_20181012_101513_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #full symetric
    #model_path = log_dir_root_dflt / 'inked_slides/inked_slides_l1smooth_20181005_191658_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked_slides/inked_slides_l1_20181006_075234_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    #model_path = log_dir_root_dflt / 'inked-slides-cmy_l1_20181026_222127_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked-slides-cmy-randclip_l1_20181026_222127_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #model_path = log_dir_root_dflt / 'R_inked-slides-cmy-randclip_l1_20181028_105836_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #inked2clean
    #model_path = log_dir_root_dflt / 'tiny-inked2real_l1_20181029_161716_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked2real_l1smooth_20181029_161716_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked2real_l2_20181029_161716_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'tiny-inked2real_bootpixl2_20181029_174432_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    #model_path = log_dir_root_dflt / 'inked2real_l1_20181029_213925_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'inked2real_l1smooth_20181029_213925_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    #model_path = log_dir_root_dflt / 'R_inked2real_l1_20181030_150946_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    model_path = log_dir_root_dflt / 'R_inked2real_bootpixl2_20181030_134605_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'R_inked2real_l2_20181030_135222_unet-ch3_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    
    
    
    scale_log = (0, 255)
    n_ch = 3
    
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    is_cmy = '-cmy' in model_path.parent.name
    #%%
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/samples/Tile004515.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/samples/Tile000341.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/samples/Tile003586.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/test_ISBI/Tile004500.jpg'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/fakes/fake.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/fakes/1.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/fakes/2.jpg'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/fakes/3.jpg'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/Tile003873.jpg'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/tiny/Tile003991.jpg'
    
    
     
    x = cv2.imread(fname, -1)[..., ::-1]/255
    #x = rgb_cmy(x)
    
    x = x.astype(np.float32)
    x = np.rollaxis(x, 2, start=0)
    
    if is_cmy:
        x = 1 - x
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    
    
    xhat_r = np.rollaxis(xhat, 0, start=3)
    xhat_r = np.clip(xhat_r, 0, 1)
    
    #xhat_r = cmy_rgb(xhat_r)
    xr = np.rollaxis(x, 0, start=3)
    
    if is_cmy:
        xhat_r = 1 - xhat_r
        xr = 1- xr
    #%%
    #xr = cmy_rgb(xr)
    
    #real_fname = '/Users/avelinojaver/OneDrive - Nexus365/inked_slides/clean/Tile001421.jpg'
    #img_real = cv2.imread(real_fname, -1)[..., ::-1].astype(np.float32)/255
    
    fig, axs = plt.subplots(1,2,sharex=True, sharey=True)
    
    axs[0].imshow(xr)#, vmin=0, vmax=1)
    axs[0].set_title('input')
    axs[1].imshow(xhat_r)#, vmin=0, vmax=1)
    axs[1].set_title('output')
    #axs[2].imshow(img_real)
    #axs[2].set_title('ground truth')
    