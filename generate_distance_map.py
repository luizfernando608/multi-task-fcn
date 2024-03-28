import os

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label
from tqdm import tqdm

from src.utils import (array2raster, check_folder, get_image_metadata,
                       print_sucess, read_tiff, read_yaml)
from logging import getLogger

logger = getLogger("__main__")

def apply_gaussian_distance_map(input_img:np.ndarray, sigma=5)->np.ndarray:
    """Apply euclidean distance transform and gaussian filter to the input image

    Parameters
    ----------
    input_img : np.ndarray
        Input image matrix with the segmentation

    Returns
    -------
    np.array
        Output image matrix with the distance map and gaussian filter applied to each segmentation component
    """

    ref = input_img.copy()
    # ref[ref>0] = 1

    # label the image as components
    label_ref = label(ref)

    # apply distance transform and gaussian filter
    new_lab = label_ref.copy()
    new_lab = distance_transform_edt(new_lab)
    new_lab = gaussian_filter(new_lab, sigma)
    

    # create the new image with the distance map
    save_lab = np.zeros(label_ref.shape)

    for component in tqdm(np.unique(label_ref[label_ref>0])):

        component_mask = label_ref==component

        save_lab[component_mask] = new_lab[component_mask]/np.max(new_lab[component_mask])
        
        
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

    img_metadata = get_image_metadata(input_image_path)

    input_img = read_tiff(input_image_path).astype('uint16')
    
    output_img = apply_gaussian_distance_map(input_img)

    
    array2raster(output_image_path, output_img, img_metadata, "float32")


if __name__ == "__main__":
    
    args = read_yaml("args.yaml")
    
    train_input_path = os.path.join(args.data_path, args.train_segmentation_file)

    train_output_path = os.path.join(args.data_path, "test", "train_distance_map.tif")
    
    check_folder(os.path.dirname(train_output_path))

    generate_distance_map(train_input_path, train_output_path)
    
    print_sucess("Distance map generated successfully")