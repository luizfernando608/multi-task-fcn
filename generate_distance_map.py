import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label

from src.utils import (array2raster, check_folder, get_image_metadata,
                       print_sucess, read_tiff, read_yaml)


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

    BLOCK_SIZE = 1000

    # convert into bool to improve label() func performance
    ref = (input_img > 0).astype("bool")

    # label the image as components
    label_ref = label(ref)

    # Get the dimensions of the image
    height, width = label_ref.shape

    # Iterate over blocks and apply distance transform
    result = np.zeros((height, width), dtype=np.float32)

    # Apply transformation to blocks with overlap
    for y in range(0, height, BLOCK_SIZE//2):

        for x in range(0, width, BLOCK_SIZE//2):
            
            # Define the boundaries of the block
            y_end = min(y + BLOCK_SIZE, height)
            x_end = min(x + BLOCK_SIZE, width)

            # Process the current block
            block = label_ref[y:y_end, x:x_end]
            
            # Apply euclidean distance transform
            block = distance_transform_edt(block)
            # Apply gaussian filter
            block = gaussian_filter(block, sigma)
            
            # Select the minimum non-zero value for each pixel
            final_block = np.where(result[y:y_end, x:x_end] > 0,
                                   np.minimum(block, result[y:y_end, x:x_end]),
                                   block)

            # store block transformed
            result[y:y_end, x:x_end] = final_block


    # create the new image with the distance map
    save_lab = np.zeros(label_ref.shape)

    for obj in np.unique(label_ref[np.nonzero(label_ref)]):
        
        # normalize the distance map
        save_lab[label_ref==obj] = result[label_ref==obj]/np.max(result[label_ref==obj])
        
    return save_lab
    



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
    
    train_input_path = args.train_segmentation_path

    train_output_path = os.path.join(args.data_path, "test", "train_distance_map.tif")
    
    check_folder(os.path.dirname(train_output_path))

    generate_distance_map(train_input_path, train_output_path)

    print_sucess("Distance map generated successfully")