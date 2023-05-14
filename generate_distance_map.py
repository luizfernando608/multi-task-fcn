#%%
import os
import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from src.utils import check_folder, array2raster, read_tiff, read_yaml, print_sucess
import pandas as pd
import matplotlib.pyplot as plt

args = read_yaml("args.yaml")


def apply_gaussian_distance_map(input_file, output_file):
    """Apply distance map with a gaussian filter to the ground truth segmentantion
    and save it as a tiff file

    Parameters
    ----------
    input_file : str
        File name of the input image in the DATA_PATH folder
    output_file : str
        Output file name to be saved in the DATA_PATH folder
    """
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



def is_generated(output_file):
    """Check if the distance map has already been generated

    Returns
    -------
    bool
        True if the distance map has already been generated, False otherwise
    """
    if os.path.exists(os.path.join(args.data_path, "before_iter")):
        return os.path.exists(os.path.join(args.data_path, "before_iter", output_file))
        
    else:
        return False

    return False



# %%
def generate_distance_map(input_image, output_image):
    if not is_generated(output_image):
        check_folder(os.path.join(args.data_path, "before_iter"))
        output_path = os.path.join(args.data_path, "before_iter", output_image)
        input_path = os.path.join(args.data_path,"segmentation", input_image)
        apply_gaussian_distance_map(input_path, output_path)
        

def generate_train_test_map():
    generate_distance_map(args.train_segmentation_file, "train_distance_map.tif")
    generate_distance_map(args.test_segmentation_file, "test_distance_map.tif")
    print_sucess("Distance map generated successfully")

