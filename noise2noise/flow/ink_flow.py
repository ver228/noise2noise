#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:27:06 2018

@author: avelinojaver
"""

import cv2
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random

_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/inked_slides/')
#_root_dir = Path.home() / 'workspace/denoising_data/inked_slides'

def rgb_cmyk(x):
    x = x.astype(np.float32)/255.
    K = 1 - x.max(axis=-1)
    N = (1 - x - K[..., None])/(1 - K[..., None]+ 1e-8)
    return np.concatenate((N, K[..., None]), axis=2)

def cmyk_rgb(x):
    K = x[..., -1]
    N = x[..., :-1]
    x_rgb = 1 - N*(1-K[..., None]) -  K[..., None]
    return x_rgb

class InkedFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 crop_size = 256,
                 samples_per_epoch = 1000
                 ):
        print(root_dir)
        root_dir = Path(root_dir)
        self.clean_dir = root_dir / 'clean'
        self.ink_dir = root_dir / 'ink'
        self.crop_size = crop_size 
        self.samples_per_epoch = samples_per_epoch
        
        self.clean_files = [x for x in self.clean_dir.glob('*.jpg') if not x.name.startswith('.')]
        self.ink_files =  [x for x in self.ink_dir.glob('*.jpg') if not x.name.startswith('.')]
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, ind):
        clean_file = random.choice(self.clean_files)
        img_clean = cv2.imread(str(clean_file), -1)[..., ::-1]
        
        clean_crop = self._augment(img_clean, self.crop_size, self.crop_size)
        clean_crop = rgb_cmyk(clean_crop)
        
        out1 = self._add_ink(clean_crop.copy())
        out2 = self._add_ink(clean_crop.copy())
        
        out1 = np.rollaxis(out1, 2, start=0)
        out2 = np.rollaxis(out2, 2, start=0)
        
        return out1, out2
        
    def _add_ink(self, clean_crop):
        ink_file = random.choice(self.ink_files)
        img_ink = cv2.imread(str(ink_file), -1)[..., ::-1]
        
        nn = random.randint(0, 3)
        
        for _ in range(nn):
            crop_x = random.randint(self.crop_size//4, self.crop_size)
            crop_y = random.randint(self.crop_size//4, self.crop_size)
            ink_crop = self._augment(img_ink, crop_x, crop_y)
            
            ink_crop = rgb_cmyk(ink_crop)
            
            w,h, _ = clean_crop.shape
            wc, hc, _ = ink_crop.shape
            ix = random.randint(0, w - wc)
            iy = random.randint(0, h - hc)
            
            clean_crop[ix:ix+wc, iy:iy+hc] += ink_crop
        clean_crop = np.clip(clean_crop, 0., 1.)
        
        return clean_crop
        
    def _augment(self, X, crop_x, crop_y):
        w,h, _ = X.shape
        ix = random.randint(0, w-crop_x)
        iy = random.randint(0, h-crop_y)
        X = X[ix:ix+crop_x, iy:iy+crop_y]
        
        #horizontal flipping
        if random.random() < 0.5:
            X = X[::-1]
            
        #vertical flipping
        if random.random() < 0.5:
            X = X[:, ::-1]
        
        return X



class InkedFlowBG(InkedFlow):
    def __getitem__(self, ind):
        clean_crop = np.zeros((self.crop_size, self.crop_size, 4), np.float32)
        
        out1 = self._add_ink(clean_crop.copy())
        
        #out1 = np.rollaxis(out1, 2, start=0)
        #out2 = np.rollaxis(out2, 2, start=0)
        
        return out1
    
    
#%%
if __name__ == '__main__':
    gen = InkedFlow()
    for _ in range(10):
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        
        for ii, x in enumerate(gen[0]):
            img = np.rollaxis(x, 0, start=3)
            img = cmyk_rgb(img)
            axs[ii].imshow(img)
    
#    #%%
#    import tqdm
#    crop_size  = 1578
#    gen = InkedFlowBG(crop_size=crop_size)
#    
#    avg_img = np.zeros((crop_size, crop_size, 4), np.float32)
#    N = 1000
#    for _ in tqdm.tqdm(range(N)):
#        avg_img += gen[0]
#    avg_img /= N
#    #%%
#    fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
#    
#    img_r = cmyk_rgb(avg_img)
#    axs.imshow(img_r)
