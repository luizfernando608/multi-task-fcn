#%%
import os
import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from src.utils import check_folder, array2raster, read_tiff, read_yaml

import matplotlib.pyplot as plt
args = read_yaml("args.yaml")


def generate_gaussian_distance_map(input_file, output_file):
    # load the reference image
    DATA_PATH = args.data_path
    input_image_path = os.path.join(DATA_PATH, input_file)
    
    reference_img = gdal.Open(input_image_path)

    ref = read_tiff(input_image_path).astype('uint16')
    ref[ref>0] = 1
    # generate the distance map
    label_ref = label(ref)
    new_lab = label_ref.copy()
    new_lab = distance_transform_edt(new_lab)
    new_lab = gaussian_filter(new_lab, sigma=5)
    save_lab = np.zeros(label_ref.shape)
    for obj in np.unique(label_ref)[1:]:
        try:
            save_lab[label_ref==obj] = new_lab[label_ref==obj]/np.max(new_lab[label_ref==obj])
        except:
            print('what')
    
    # save the distance map
    output_image_path = os.path.join(DATA_PATH, output_file)
    array2raster(output_image_path, reference_img, save_lab, 'Float32')



#%%
input_train_file = "samples_A1_train2tif.tif"
input_test_file = "samples_A1_test2tif.tif"
output_train_file = "train_depth.tif"
output_test_file = "test_depth.tif"

generate_gaussian_distance_map(input_train_file, output_train_file)
generate_gaussian_distance_map(input_test_file, output_test_file)

# %%
