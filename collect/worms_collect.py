#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""

from pathlib import Path
import tqdm
import tables
import cv2

if __name__ == '__main__':
    src_root_dir = Path.home() / 'workspace/WormData/screenings/'
    save_root_dir = Path.home() / 'workspace/WormData/full_images/'
    
    fnames = [x for x in src_root_dir.rglob('*.hdf5') if 'MaskedVideos' in str(x)]
    
#%%
    for fname in tqdm.tqdm(fnames):
        with tables.File(fname, 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
        
        new_dname = Path(str(fname.parent / fname.stem).replace(str(src_root_dir), str(save_root_dir)))
        new_dname.mkdir(parents=True, exist_ok=True)
        
        for ii, img in enumerate(imgs):
            save_name = new_dname / '{}.tif'.format(ii)
            cv2.imwrite( str(save_name), img)
        