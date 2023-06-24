#%%
import os
import numpy as np

from osgeo import gdal

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label

from src.utils import check_folder, array2raster, read_tiff, read_yaml, print_sucess




def apply_gaussian_distance_map(input_img:np.array)->np.array:
    """Apply euclidean distance transform and gaussian filter to the input image

    Parameters
    ----------
    input_img : np.array
        Input image matrix with the segmentation

    Returns
    -------
    np.array
        Output image matrix with the distance map and gaussian filter applied to each segmentation component
    """

    ref = input_img.copy()
    ref[ref>0] = 1

    # label the image as components
    label_ref = label(ref)

    # apply distance transform and gaussian filter
    new_lab = label_ref.copy()
    new_lab = distance_transform_edt(new_lab)
    new_lab = gaussian_filter(new_lab, sigma=5)
    

    # create the new image with the distance map
    save_lab = np.zeros(label_ref.shape)

    for obj in np.unique(label_ref)[1:]:
        # normalize the distance map
        save_lab[label_ref==obj] = new_lab[label_ref==obj]/np.max(new_lab[label_ref==obj])
        
    return save_lab
    



# %%
def generate_distance_map(input_image_path:str, output_image_path:str):
    """Generate the distance map from the input image and save it in the output path
    
    Parameters
    ----------
    input_image_path : str
        Path to the input image
    
    output_image_path : str
        Path to save the output image
    """
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError("Input image not found")

    if not os.path.exists(os.path.dirname(output_image_path)):
        raise FileNotFoundError("Output folder not found")

    img_lazy_load = gdal.Open(input_image_path)

    input_img = read_tiff(input_image_path).astype('uint16')
    
    output_img = apply_gaussian_distance_map(input_img)

    array2raster(output_image_path, img_lazy_load, output_img, 'Float32')


if __name__ == "__main__":
    
    args = read_yaml("args.yaml")
    
    train_input_path = os.path.join(args.data_path, args.train_segmentation_file)

    train_output_path = os.path.join(args.data_path, "test", "train_distance_map.tif")
    
    check_folder(os.path.dirname(train_output_path))

    generate_distance_map(train_input_path, train_output_path)
    
    print_sucess("Distance map generated successfully")