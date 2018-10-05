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
    n_ch  = 1
    #model_path = log_dir_root_dflt / 'synthetic_20x_l1_20180930_175913_unet_adam_lr0.0001_wd0.0_batch8' / 'checkpoint.pth.tar'
    model_path = log_dir_root_dflt / 'microglia_synthetic_l1smooth_20181003_161017_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    
    scale_log = (0, 16)
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/N - 6(fld 1- time 1 - 22858 ms).tif'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/J - 7(fld 1- time 1 - 35958 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/L - 8(fld 1- time 1 - 71667 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180815_MicVid_20X_Stills/180815_MicVid_20X_Stills_20X_Still_InjectionWells_2/J - 8(fld 8).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/B - 9(fld 1 z 1- time 1 - 109319 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/B - 6(fld 1 z 3- time 1 - 684 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/D - 8(fld 1 z 2- time 1 - 59803 ms).tif'
    
    #blurred
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_ZTest_1/H - 10(fld 1 z 10- time 3 - 50106 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_ZTest_1/H - 10(fld 1 z 8- time 3 - 50106 ms).tif'
    
    # old
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/old_videos/Raw/Beacon-1 unst/Scene1Interval001_RFP.png'
    #scale_log = (0, 8)
    
    img = cv2.imread(fname, -1)[..., ::-1]
    img = img[None]
    
    x = img.astype(np.float32)
    
    x = np.log(x+1)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    #%%
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(xr)#, vmin=0, vmax=1)
    axs[1].imshow(xhat)#, vmin=0, vmax=1)
    axs[2].imshow(xhat>0.3)
    