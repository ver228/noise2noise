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
from noise2noise.flow import MNISTFashionFlow

import torch
import numpy as np
import cv2
#%%
if __name__ == '__main__':
    n_ch  = 1
    #mnist-fg-fix_l1_20190126_092811_unet_adam_lr1e-05_wd0.0_batch12
    #
    #mnist-bg-fix_l2_20190125_220558_unet_adam_lr1e-05_wd0.0_batch12
    
    #bn = 'mnist-fg-fix_l2_20190125_215745_unet_adam_lr1e-05_wd0.0_batch15' 
    #bn = 'mnist-fg-fix_l1smooth_20190127_000440_unet_adam_lr1e-05_wd0.0_batch12'
    #bn = 'mnist-fg-fix_l1_20190126_093344_unet_adam_lr1e-05_wd0.0_batch12'
    
    #bn = 'mnist-fg-fix_l1_20190126_093344_unet_adam_lr1e-05_wd0.0_batch12'
    #bn = 'mnist-bg-fix_l2_20190125_220558_unet_adam_lr1e-05_wd0.0_batch12'
    
    bn = 'mnist-fg-fix-v1_l2_20190128_165524_unet_adam_lr1e-05_wd0.0_batch48'
    #bn = 'mnist-fg-fix-v2_l2_20190128_163917_unet_adam_lr1e-05_wd0.0_batch48'
    
    model_path = log_dir_root_dflt / bn / 'checkpoint.pth.tar'
    
    
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    
#    argkws = dict(output_size = 256,
#                         epoch_size = 10,
#                         bg_n_range = (5, 25),
#                         int_range = (1., 1.),
#                         max_rotation = 45,
#                         is_v_flip = False)
    #argkws = dict(output_size = 256)
    
    gen = MNISTFashionFlow(is_separate = True, fg_n_range = (1, 5), **argkws)
    gen.test()
    #%%
    import tqdm
    from skimage.measure import compare_ssim
    
    all_J = []
    all_ssim = []
    #%%
    for ibatch, (x1,x2) in tqdm.tqdm(enumerate(gen)):
        
        with torch.no_grad():
            xin = x1+x2
            X = torch.from_numpy(xin[None])
            Xhat = model(X)
            
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        xin = xin.squeeze()
        xout = Xhat.squeeze().detach().numpy()
        
        th_in = x2>0.2
        th_out = xout>0.2
        
        U = (th_in | th_out)
        I = (th_in & th_out)
        
        all_J.append((I.sum(), U.sum()))
        
        
        ssim = compare_ssim(x2, xout)
        all_ssim.append(ssim)
        
        if False:
            
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(2,3,sharex=True, sharey=True, figsize = (14, 8))
            axs[0][0].imshow(x1, cmap='gray', vmin=0, vmax=1)
            axs[0][1].imshow(x2, cmap='gray', vmin=0, vmax=1)
            axs[0][2].imshow(th_in)
            
            
            
            axs[1][0].imshow(xin, cmap='gray', vmin=0, vmax=1)
            axs[1][1].imshow(xout, cmap='gray', vmin=0, vmax=1)
            
            axs[1][2].imshow(th_out)
            axs[1][2].set_title(I.sum()/U.sum())
            #%%
        
        if ibatch >= 100:
            break
    #%%    
    I, U = map(np.sum, zip(*all_J))
    mIOU = I/U
    
    print(mIOU, bn)
    
    #%%
    #from skimage.measure import compare_ssim
    
    
    
    