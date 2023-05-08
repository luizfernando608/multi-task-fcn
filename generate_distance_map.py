#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:17:17 2020

@author: laura
"""

import argparse
import os
import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from src.utils import check_folder, array2raster, read_tiff

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/ITC_ID.tif', help="Path of the gt dataset")


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    refrence_img = gdal.Open(args.gt_dir)
    ref = read_tiff(args.gt_dir).astype('uint16')
    
    new_lab = ref.copy()
    new_lab = distance_transform_edt(new_lab)
    new_lab = gaussian_filter(new_lab, sigma=5)
    
    save_lab = np.zeros(ref.shape)

    for obj in np.unique(ref)[1:]:
        save_lab[ref==obj] = (new_lab[ref==obj]-np.min(new_lab[ref==obj]))/(np.max(new_lab[ref==obj])-np.min(new_lab[ref==obj]))
    
    array2raster(os.path.join('D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation','train_depth.tif'), refrence_img, save_lab, 'Float32')
    
    
    

