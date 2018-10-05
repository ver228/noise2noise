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
    model_path = log_dir_root_dflt / 'worms_l1smooth_20181003_150642_unet_adam_lr0.0001_wd0.0_batch16' / 'checkpoint.pth.tar'
    scale_log = (0, 255)
    
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Drug_Screening/MaskedVideos/MK_Screening_Amisulpride_Chlopromazine_CB4856_240817/CB4856_worms10_Amisulpride_100_Set3_Pos4_Ch1_24082017_155105/0.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Lidia/MaskedVideos/Optogenetics-day1/AQ2050-EtOH_Set1_Ch1_18072017_163921/15.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Pratheeban/First_Set/MaskedVideos/Old_Adult/16_07_14/S3_ELA_1.0_Ch1_14072016_184723/1.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Serena_WT_Screening/MaskedVideos/agg_15.2_180304/15.2_3_da609_Set0_Pos0_Ch5_04032018_120310/10.tif'
    #fname = '/Volumes/rescomp1/data/denoising_data/c_elegans/test/Drug_Screening/MaskedVideos/MK_Amisulpride_Chlopromazine_230817_N2/N2_worms10_Chlopromazine_10_Set4_Pos5_Ch2_23082017_190422/4.tif'
    
    fname = '/Users/avelinojaver/OneDrive - Imperial College London/documents/papers_in_progress/paper_tierpsy_tracker/figures_data/different_setups/pycelegans/RawVideos/NT00029_cam1_W001_110722_1649_cam1_M00001/NT00029_cam1_W001_110722_1649_cam1_M00001_frame00000_0000000.jpg'
    
    img = cv2.imread(fname, -1)
    cv2.resize(img, dsize = tuple([x//4 for x in img.shape]))
    
    img = img[None]
    
    #%%
#    import tables
#    #fname = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/different_animals/worm_motel/MaskedVideos/Position5_Ch2_12012017_100957_s.hdf5'
#    fname = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/test_5/CSTCTest_Ch1_18112015_075624.hdf5'
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
    
    axs[0].imshow(xr)#, vmin=0, vmax=1)
    axs[1].imshow(xhat)#, vmin=0, vmax=1)
    axs[2].imshow((xhat - xr))
    #%%
    
    
#    xhat_l = np.exp(xhat*(scale_log[1] - scale_log[0]) + scale_log[0])
#    x_l = np.exp(x*(scale_log[1] - scale_log[0]) + scale_log[0])
#    
#    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
#    axs[0].imshow(x_l)
#    #xhat[xhat<0.301] = 0.301
#    axs[1].imshow(xhat_l)
#    axs[2].imshow(xhat_l - x_l)
    