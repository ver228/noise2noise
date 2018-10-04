#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
from pathlib import Path
import random
import cv2
import numpy as np

from torch.utils.data import Dataset 

_root_dir = Path.home() / 'workspace/denoising_data/microglia/syntetic_data'

class SyntheticFluoFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 bgnd_path_size = (512, 512),
                 n_cells_per_crop = 4,
                 int_factor = (0.01, 1.0),
                 epoch_size = 2000,
                 scale_log = (0, 16)
                 ):
        
        print(root_dir)
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.bgnd_path_size = bgnd_path_size
        self.n_cells_per_crop = n_cells_per_crop
        self.int_factor = int_factor
        self.epoch_size = epoch_size
        self.scale_log = scale_log
        
        cell_root_dir = root_dir / 'cell_images'
        bgnd_root_dir = root_dir / 'bgnd_images'
        
        cell_files = list(cell_root_dir.rglob('*.tif'))
        bgnd_files = list(bgnd_root_dir.rglob('*.tif'))
        
        dd = str(cell_root_dir) + '/'
        cell_files_d = {}
        for x in cell_files:
            path_r = str(x.parent).replace(dd, '')
            
            if not path_r in cell_files_d:
                cell_files_d[path_r] = {}
            
            bn = '_'.join([x for x in x.name.split('_') if x[0] != 't'])
            if not bn in cell_files_d[path_r]:
                cell_files_d[path_r][bn] = []
            
            cell_files_d[path_r][bn].append(x)
        
        for path_r in cell_files_d:
            for bn in cell_files_d[path_r]:
                cell_files_d[path_r][bn] =  sorted(cell_files_d[path_r][bn])
            
            cell_files_d[path_r] = list(cell_files_d[path_r].values())
        
        dd = str(bgnd_root_dir) + '/'
        bgnd_files_d = {}
        for x in bgnd_files:
            path_r = str(x.parent).replace(dd, '')
            
            if not path_r in bgnd_files_d:
                bgnd_files_d[path_r] = []
            bgnd_files_d[path_r].append(x)
    
    
        self.available_paths = list(bgnd_files_d.keys())
        self.bgnd_files_d = bgnd_files_d
        self.cell_files_d = cell_files_d
        
    def _get_random_bgnd(self, path_r):
        bgnd_file = random.choice(self.bgnd_files_d[path_r])
        img_bgnd = cv2.imread(str(bgnd_file), -1)
        xi = random.randint(0, img_bgnd.shape[0] - self.bgnd_path_size[0])
        yi = random.randint(0, img_bgnd.shape[1] - self.bgnd_path_size[1])
        crop_bgnd = img_bgnd[xi:xi + self.bgnd_path_size[0], yi:yi + self.bgnd_path_size[1]]
        crop_bgnd = crop_bgnd.astype(np.float32)
        return crop_bgnd
    
    def _log_transform(self, x):
        x = np.log(x+1)
        x = (x-self.scale_log[0])/(self.scale_log[1]-self.scale_log[0])
        return x
    
    def _sample(self, path_r):
        #%%
        cell_imgs = []
        for _ in range(random.randint(0, self.n_cells_per_crop)):
            fnames = random.choice(self.cell_files_d[path_r])
            ind = random.randint(0, max(0, len(fnames)-2))
            
            img_pairs = []
            
            _flipv = random.random() >= 0.5
            _fliph = random.random() >= 0.5
            #_int_fac = random.uniform(*self.int_factor)
            _int_fac = np.exp(random.uniform(*np.log(self.int_factor)))
            for ii in [ind, min(len(fnames)-1, ind+1)]:
                cell_file = fnames[ii]
                cc = cv2.imread(str(cell_file), -1).astype(np.float32)
                base_int = min(np.median(cc[0,:]), np.median(cc[:, 0]))
                cc -= base_int
                
                #random flips
                if _fliph:
                    cc = cc[::-1]
                if _flipv:
                    cc = cc[:, ::-1]
                
                cc = cc[:self.bgnd_path_size[0], :self.bgnd_path_size[1]] #crop in case it is larger than the crop
                cc *= _int_fac
            
                img_pairs.append(cc)
                
            #shift order (either before/after or after/before)
            if random.random() >= 0.5:
                img_pairs = img_pairs[::-1]
            
            
            frac_cc = [int(round(x*0.9)) for x in cc.shape]
            
            max_ind_x = self.bgnd_path_size[0] - cc.shape[0]
            max_ind_y = self.bgnd_path_size[1] - cc.shape[1]
            xi = random.randint(-frac_cc[0], max_ind_x + frac_cc[0])
            yi = random.randint(-frac_cc[1], max_ind_y + frac_cc[1])
            
            if xi < 0:
                img_pairs = [cc[abs(xi):] for cc in img_pairs]
                xi = 0
            
            if yi < 0:
                img_pairs = [cc[:, abs(yi):] for cc in img_pairs]
                yi = 0
            
            
            if xi > max_ind_x:
                ii = max_ind_x-xi
                img_pairs = [cc[:ii] for cc in img_pairs]
            
            if yi > max_ind_y:
                ii = max_ind_y - yi
                img_pairs = [cc[:, :ii] for cc in img_pairs]
            
            
            coords = (xi, yi)
            cell_imgs.append((coords, img_pairs))
        
        
        syntethic_pairs = []
        for ii in range(2):
            crop_bgnd = self._get_random_bgnd(path_r)
            
            _int_fac = random.uniform(0.9, 1.1)
            #random flips
            if random.random() >= 0.5:
                crop_bgnd = crop_bgnd[::-1]
            if random.random() >= 0.5:
                crop_bgnd = crop_bgnd[:, ::-1]
            
            crop_bgnd *= _int_fac
            
            for (xi,yi), img_pairs in cell_imgs:
                cc = img_pairs[ii]
                crop_bgnd[xi:xi+cc.shape[0], yi:yi+cc.shape[1]] += cc
                crop_bgnd[xi:xi+cc.shape[0], yi:yi+cc.shape[1]] += cc
            
            crop_bgnd = np.clip(crop_bgnd, 0, 2**16-1)
            syntethic_pairs.append(self._log_transform(crop_bgnd))
        
        return syntethic_pairs
    
    def __len__(self):
        return self.epoch_size
    
    
    def __getitem__(self, ind):
        path_r = random.choice(self.available_paths)
        return [x[None] for x in self._sample(path_r)]
#%%
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    batch_size = 16
    
    gen = SyntheticFluoFlow()
    loader = DataLoader(gen, batch_size=batch_size, shuffle=True)
    
    for Xin, Xout in loader:
        break
    
    for ii in range(batch_size):
        xin = Xin[ii].squeeze().detach().numpy()
        xout = Xout[ii].squeeze().detach().numpy()
        
        fig, axs = plt.subplots(1,2, figsize = (12, 8), sharex=True, sharey=True)
        
        vmax = max(xout.max(), xin.max())
        vmin = min(xout.min(), xin.min())
        axs[0].imshow(xin, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
        axs[1].imshow(xout, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
        axs[0].axis('off')
        axs[1].axis('off')
        
        
    
    
    