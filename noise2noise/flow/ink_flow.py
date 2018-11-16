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

#_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/inked_slides/')
_root_dir = Path.home() / 'workspace/denoising_data/inked_slides'

#def rgb_cmyk(x):
#    x = x.astype(np.float32)
#    K = 1 - x.max(axis=-1)
#    N = (1 - x - K[..., None])/(1 - K[..., None]+ 1e-8)
#    return np.concatenate((N, K[..., None]), axis=2)
#
#def cmyk_rgb(x):
#    K = x[..., -1]
#    N = x[..., :-1]
#    x_rgb = 1 - N*(1-K[..., None]) -  K[..., None]
#    return x_rgb

def rgb_cmy(x):
    x = x.astype(np.float32)
    return 1-x

def cmy_rgb(x):
    x = x.astype(np.float32)
    return 1-x


class InkedFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 crop_size = 256,
                 samples_per_epoch = 1000,
                 is_tiny = False,
                 is_return_rgb = True,
                 is_preload = False,
                 is_clipped = False,
                 is_random_clipped = False,
                 is_clean_output = False,
                 is_symetric_ink = True
                 ):
        print(root_dir)
        root_dir = Path(root_dir)
        
        if is_tiny:
            root_dir = root_dir / 'tiny'
        
        self.clean_dir = root_dir / 'clean'
        self.ink_dir = root_dir / 'ink'
        self.crop_size = crop_size 
        self.samples_per_epoch = samples_per_epoch
        self.is_tiny = is_tiny
        self.is_return_rgb = is_return_rgb
        self.is_preload = is_preload
        self.is_clipped = is_clipped
        self.is_random_clipped = is_random_clipped
        self.is_clean_output = is_clean_output
        self.is_symetric_ink = is_symetric_ink
        
        self._index = 0
        
        self.clean_files = [x for x in self.clean_dir.glob('*.jpg') if not x.name.startswith('.')]
        self.ink_files =  [x for x in self.ink_dir.glob('*.jpg') if not x.name.startswith('.')]
        
        #if self.is_tiny:
        #    self.img_clean = cv2.imread(str(self.clean_files[0]), -1)[..., ::-1]/255
        #    self.img_ink = cv2.imread(str(self.ink_files[0]), -1)[..., ::-1]/255
        
        
        if self.is_preload:
            self.imgs_clean = [cv2.imread(str(x), -1)[..., ::-1]/255 for x in self.clean_files]
            self.imgs_ink = [cv2.imread(str(x), -1)[..., ::-1]/255 for x in self.ink_files]
            
            
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, ind):
        if self.is_preload:
            img_clean = random.choice(self.imgs_clean)
            
        else:
            clean_file = random.choice(self.clean_files)
            img_clean = cv2.imread(str(clean_file), -1)[..., ::-1]/255
            
        
        clean_crop = self._augment(img_clean, self.crop_size, self.crop_size)
        clean_crop = rgb_cmy(clean_crop)
        
        out1 = self._add_ink(clean_crop.copy())
        
        if self.is_clean_output:
            out2 = clean_crop.copy()
        else:
            out2 = self._add_ink(clean_crop.copy())
        
        
        if self.is_return_rgb:
            out1 = cmy_rgb(out1)
            out2 = cmy_rgb(out2)
        
        
        out1 = np.rollaxis(out1, 2, start=0)
        out2 = np.rollaxis(out2, 2, start=0)
        
        return out1, out2
    
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
            
        self._index += 1
        return self[0]
    
    def _add_ink(self, clean_crop):
        
        if self.is_preload:
            img_ink = random.choice(self.imgs_ink)
        else:
            ink_file = random.choice(self.ink_files)
            img_ink = cv2.imread(str(ink_file), -1)[..., ::-1]/255
        
        nn = random.randint(1, 1)
        
        for _ in range(nn):
            crop_x = random.randint(self.crop_size//2, self.crop_size)
            crop_y = random.randint(self.crop_size//2, self.crop_size)
            ink_crop = self._augment(img_ink, crop_x, crop_y)
            
            ink_crop = rgb_cmy(ink_crop)
            
            w,h, _ = clean_crop.shape
            wc, hc, _ = ink_crop.shape
            ix = random.randint(0, w - wc)
            iy = random.randint(0, h - hc)
            
            rand_factor = 0.6*np.random.random_sample(3)+0.7
            ch_l = [0,1,2]
            random.shuffle(ch_l)
            ink_crop = ink_crop[..., ch_l]*rand_factor[None, None]
            
            if self.is_symetric_ink:
                factor = random.choice((1., -1.))
            else:
                factor = 1.
                
            clean_crop[ix:ix+wc, iy:iy+hc] += factor*ink_crop
        
        if self.is_random_clipped:
            is_clipped = random.random() < 0.5
        else:
            is_clipped = self.is_clipped
            
        if is_clipped:
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
        clean_crop = np.full((self.crop_size, self.crop_size, 4), 0.5, np.float32)
        
        out1 = self._add_ink(clean_crop.copy())
        
        #out1 = np.rollaxis(out1, 2, start=0)
        #out2 = np.rollaxis(out2, 2, start=0)
        
        return out1
    
    
#%%
if __name__ == '__main__':
    import tqdm
#    gen = InkedFlow(samples_per_epoch = 250, is_tiny=True, 
#                    is_random_clipped = True, is_preload=True, 
#                    is_clipped=False, is_return_rgb=False)
    
    #gen = InkedFlow(samples_per_epoch = 250, is_tiny=False, is_preload=False, is_clipped=True)
    
    gen = InkedFlow(samples_per_epoch = 250, is_tiny=True, 
                    is_preload=True, 
                    is_clean_output = True, is_symetric_ink=False, 
                    is_clipped=True, is_return_rgb=True)
    
        
    for _ in range(10):
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        
        for ii, x in enumerate(gen[0]):
            img = np.rollaxis(x, 0, start=3)
            #img = cmyk_rgb(img)
            
            #bot, top = img.min(), img.max()
            #img = (img-bot)/(top-bot)
            
            axs[ii].imshow(img)
            
    
    #%%
    
    edges = np.arange(-256, 255*2+1) + 0.5
    count_samples = np.zeros((3, len(edges)-1))
    for dat in tqdm.tqdm(gen):
        for x in dat:
            for ii in range(3):
                xi = x[ii]
                
                cc, _ = np.histogram((xi*255).flat, bins=edges)
                count_samples[ii] += cc
    
    #%%
    plt.plot(edges[1:], count_samples.T)
    #%%
#    data_types = ['ink', 'clean', 'samples']
#    counts = np.zeros((len(data_types), 3, 256))
#    for ii, dd in enumerate(data_types):
#        dname = _root_dir / dd
#        
#        fnames = list(dname.glob('*.jpg'))
#        
#        
#        for fname in tqdm.tqdm(fnames):
#            img = cv2.imread(str(fname), -1)
#            if img is None:
#                continue
#            
#            for ic in range(3):
#                counts[ii, ic] += np.bincount(img[..., ic].flat, minlength=256)
#    #%%
#    fig, axs = plt.subplots(3, 1, sharex=True)
#    for ii, ax in enumerate(axs):
#        hh = counts[:, ii]
#        for d, h in zip(data_types, hh):
#            h = h / h.sum()
#            ax.plot(h, label=d)
#        
#        
#        cc = count_samples[ii]
#        cc = cc / cc.sum()
#        ax.plot(edges[1:], cc, label='augment')
#        ax.legend()
        
        
        #%%
        
#    fig, axs = plt.subplots(3, 1, sharex=True)
#    for ax, cc in zip(axs, count_samples):
#        h = cc / cc.sum()
#        ax.plot(edges[1:], cc)
    #%%
#        hh = counts[:, ii]
#        for d, h in zip(data_types, hh):
#            h = h / h.sum()
#            ax.plot(h, label=d)
#        ax.legend()
#    plt.plot(count_samples[0])
        
        
#%%
#    import fire
#    fire.Fire(get_histograms)

    
    #%%
#    import tqdm
#    crop_size  = 128
#    gen = InkedFlowBG(crop_size=crop_size)
#    
#    avg_img = np.zeros((crop_size, crop_size, 4), np.float32)
#    N = 1000
#    for _ in tqdm.tqdm(range(N)):
#        avg_img += gen[0]
#    avg_img /= N
#    
#    fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
#    
#    img_r = cmyk_rgb(avg_img)
#    axs.imshow(img_r)
