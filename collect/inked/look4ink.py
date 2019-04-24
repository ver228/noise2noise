#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:52:17 2019

@author: avelinojaver
"""
from pathlib import Path
import cv2

#%%
if __name__ == '__main__':
    import tqdm
    
    #root_dir = Path.home() / 'workspace/histology_inked/raw/47502_16_601169_RP_6M'
    root_dir = Path.home() / 'workspace/histology_inked/raw/47502_16_601169_RP_6N'
    
    fnames = list(root_dir.glob('*.jpg'))
    
    with open(root_dir / 'tilesWithInk.txt') as fid:
        inked_names = fid.read().split('\n')
        
    #%%
    inked_files = [x for x in fnames if x.name in inked_names]
    clean_files = [x for x in fnames if not x.name in inked_names]
    #%% 
    
    img_avgs = []
    for ifname, fname in enumerate(tqdm.tqdm(inked_files)):
        img = cv2.imread(str(fname), -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        plt.figure()
        plt.imshow(img[:, :, ::-1])
        
        
        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
        for ii, (ax, tt) in enumerate(zip(axs[0], 'BGR')):
            ax.imshow(img[:, :, ii], cmap='gray')
            ax.set_title(tt)
        for ii, (ax, tt) in enumerate(zip(axs[1], 'HSV')):
             ax.imshow(hsv[:, :, ii], cmap='gray')
             ax.set_title(tt)
        
        if ifname > 10:
            break    
    #%%
    #%%
    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hsv_a = hsv.copy()
    hsv_a[:, :, 2] = hsv[:, :, 2]*2
    img_rgb = cv2.cvtColor(hsv_a, cv2.COLOR_HSV2RGB)
    
    axs[0].imshow(img[:, :, ::-1])
    axs[1].imshow(img_rgb)
    
    
    
    