"""
Created on Wed Feb  5 10:17:17 2020
@author: laura
"""
import os
import sys
import glob
import logging
import argparse
import subprocess
import numpy as np

from osgeo import ogr
from osgeo import gdal
from skimage.measure import label
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from src.utils import check_folder, array2raster, read_tiff
from coloredlogs import ColoredFormatter
from skimage.morphology import dilation, disk 
from sklearn.model_selection import StratifiedKFold
from skimage.measure import label, regionprops, regionprops_table

from src.utils import read_yaml

def make_checkerboard(n_rows, n_columns, square_size):

    n_rows_, n_columns_ = int(n_rows/square_size + 1), int(n_columns/square_size + 1)
    rows_grid, columns_grid = np.meshgrid(range(n_rows_), range(n_columns_), indexing='ij')
    high_res_checkerboard = np.mod(rows_grid, 2) + np.mod(columns_grid, 2) == 1
    square = np.ones((square_size,square_size))
    checkerboard = np.kron(high_res_checkerboard, square)[:n_rows,:n_columns]

    return checkerboard

def main(args):
    """
    :param args:
    :return:
    """
     
    raster_gt = read_tiff(args.raster_gt) 
    itc_id = label(raster_gt)
    raster_src = gdal.Open(args.raster_gt)
    ################################# Random Seelection ########################################
    
    region_props = regionprops_table(itc_id, properties=("label","centroid"))
    unique_itc = len(np.unique(itc_id[itc_id!=0]))
    lab_x = np.zeros(unique_itc)
    lab_y = np.zeros(unique_itc)
    for j in range(unique_itc):
        lab_x[j] = region_props["label"][j]
        x,y = region_props["centroid-0"][j],region_props["centroid-1"][j]
        lab_y[j] = raster_gt[int(x),int(y)]
        

    skf = StratifiedKFold(n_splits=4, random_state=31, shuffle=True)
    cont = 0
    for train_index, test_index in skf.split(lab_x, lab_y):
        itc_train, itc_test = lab_x[train_index], lab_x[test_index]
        
        tmp_gt_test = np.zeros(raster_gt.shape)
        tmp_gt_tr = raster_gt.copy()
        for i in np.unique(itc_test):            
            tmp_gt_test[itc_id==i] = raster_gt[itc_id==i]
            tmp_gt_tr[itc_id==i] = 0
    
        array2raster(os.path.join(args.out_path, 'random_train_{}.TIF'.format(cont)), raster_src, tmp_gt_tr, 'Byte')
        array2raster(os.path.join(args.out_path, 'random_test_{}.TIF'.format(cont)), raster_src, tmp_gt_test, 'Byte')

        cont+=1 
        
    ################################# checkerboard ########################################
    

    raster_mask = make_checkerboard(raster_gt.shape[0], 
                                    raster_gt.shape[1], 
                                    1024)
    unique_mask = np.unique(raster_mask)
    
    array2raster(os.path.join(args.out_path, 'checkboard.TIF'), raster_src, unique_mask, 'Byte')
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Binarizes images according to shapefile attributes')
    # parser.add_argument('--raster_gt', default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/ITC_CLASS.tif',
    #                     help="Path of the refence raster image")
    # parser.add_argument('--itc_id', default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/ITC_ID.tif',
    #                     help="Path of the ITC unique id raster image")
    # parser.add_argument('--mask_cluster', default='D:/Projects/PUC-PoC/data_new/amazonas/isa_upa/clustered_shape.TIF',
    #                     help="Path of the mask image for cluster split")
    # parser.add_argument('--out_path', default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation',
    #                     help="Path to save the images")

    # args = parser.parse_args()
    args = read_yaml("args.yaml")

    main(args)