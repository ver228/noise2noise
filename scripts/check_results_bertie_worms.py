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
    model_path = log_dir_root_dflt / 'bertie_worms_l1_20181210_162006_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    #model_path = log_dir_root_dflt / 'bertie_worms_l2_20181210_163441_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    scale_log = (0, 255)
    
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Drug_Screening/MaskedVideos/MK_Screening_Amisulpride_Chlopromazine_CB4856_240817/CB4856_worms10_Amisulpride_100_Set3_Pos4_Ch1_24082017_155105/0.tif'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/2017_07_01/Bertie_2017_07_01_1/CX11314/CX11314_Ch1_01072017_093003/1.tif'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/2017_07_01/Bertie_2017_07_01_1/CX11314/CX11314_Ch2_01072017_093003/4.tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/2017_07_01/Bertie_2017_07_01_1/N2H/N2H_Ch2_01072017_100340/4.tif'
    
    img = cv2.imread(fname, -1)
    #cv2.resize(img, dsize = tuple([x//4 for x in img.shape]))
    img = img[None]
    
    #%%
#    fname = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/different_animals/worm_motel/MaskedVideos/Position5_Ch2_12012017_100957_s.hdf5'
#    #fname = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/test_5/CSTCTest_Ch1_18112015_075624.hdf5'
#    import tables
#    with tables.File(fname, 'r') as fid:
#        img = fid.get_node('/full_data')[0]
#        img = img[None]
    #%%
    
    
    x = img.astype(np.float32)
    
    #x = np.log(x+1)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    
    #%%
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
    axs[0].set_title('Input')
    axs[1].imshow(xhat, cmap='gray')#, vmin=0, vmax=1)
    axs[1].set_title('Output')
    
    rr = (xr - xhat)
    
    axs[2].imshow(rr)
    axs[2].set_title('Input  - Output')
    for ax in axs:
        ax.axis('off')
        
    #%%
    
    
#    ix, iy = np.unravel_index(np.argmax(rr), rr.shape)
#    ss = 64
#    roi = rr[ix-ss:ix+ss, iy-ss:iy+ss]
#    
#    
#    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
#    
#    roi_m = roi.copy()
#    for _ in range(3):
#        roi_m = cv2.medianBlur(roi_m, 3)
#    
#    roi_l = cv2.Laplacian(roi_m,cv2.CV_32F)
#    axs[0].imshow(roi)
#    axs[1].imshow(roi_m)
#    axs[2].imshow(roi_l)
    
    
    