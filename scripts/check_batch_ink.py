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
import tqdm
#%%
if __name__ == '__main__':
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model_dirs = [
            'inked_slides/tiny/inked_slides_l1_20181012_101949_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'inked_slides/tiny/R_inked-slides-clipped_l1_20181012_111351_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            
            'inked_slides/inked-slides-clipped_l1smooth_20181005_173844_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'inked_slides/inked-slides-clipped_l1_20181012_101513_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            
            'inked_slides/inked_slides_l1smooth_20181005_191658_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'inked_slides/inked_slides_l1_20181006_075234_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            
            
            'R_inked-slides-cmy-randclip_l1_20181028_105836_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'R_inked-slides-cmy-randclip_l1_20181028_110431_unet-ch3_adam_lr1e-05_wd0.0_batch16',
            'R_inked-slides-cmy_l1_20181028_105836_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'R_inked-slides-cmy_l1_20181028_110431_unet-ch3_adam_lr1e-05_wd0.0_batch16',
            'inked-slides-cmy-randclip_l1_20181026_222127_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            'inked-slides-cmy_l1_20181026_222127_unet-ch3_adam_lr0.0001_wd0.0_batch16',
            ]
   
    
    scale_log = (0, 255)
    n_ch = 3
    
    #files2check = (Path.home() / 'workspace/denoising_data/inked_slides/samples/').glob('*.jpg')
    #save_root = log_dir_root_dflt / '_outputs'
    
    
    #files2check = (Path.home() / 'workspace/denoising_data/inked_slides/test_ISBI').glob('*.jpg')
    #save_root = log_dir_root_dflt / '_outputs_test_ISBI'
    
    files2check = (Path.home() / 'workspace/denoising_data/inked_slides/fakes').glob('*.jpg')
    save_root = log_dir_root_dflt / '_outputs_fakes'
    
    
    files2check = [x for x in files2check if not x.name.startswith('.')]
    print(files2check)
    
    for model_dir in tqdm.tqdm(model_dirs):
        is_cmy = '-cmy' in model_dir
        
        model_path =  log_dir_root_dflt / model_dir  / 'checkpoint.pth.tar'
        assert model_path.exists()
        
        save_dir = save_root / model_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        
        
        
        model = UNet(n_channels = n_ch, n_classes = n_ch)
        state = torch.load(str(model_path), map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model = model.to(device)
        
        model.eval()
        
        
        
        for fname in tqdm.tqdm(files2check):
            x = cv2.imread(str(fname), -1)[..., ::-1]/255
            x = x.astype(np.float32)
            x = np.rollaxis(x, 2, start=0)
            
            if is_cmy:
                x = 1 - x
            
            with torch.no_grad():
                X = torch.from_numpy(x[None])
                X = X.to(device)
                Xhat = model(X)
            
            xhat = Xhat.squeeze().cpu().detach().numpy()
    
            xhat_r = np.rollaxis(xhat, 0, start=3)
            xhat_r = np.clip(xhat_r, 0, 1)
            
            if is_cmy:
                xhat_r = 1 - xhat_r
            
            save_name = save_dir / ('out_' + fname.stem + '.png')
            cv2.imwrite(str(save_name), (xhat_r[..., ::-1]*255).astype(np.uint8))
            

        