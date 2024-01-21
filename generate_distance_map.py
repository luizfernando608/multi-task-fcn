import os
from os.path import dirname, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label

from src.utils import (array2raster, check_folder, get_image_metadata,
                       print_sucess, read_tiff, read_yaml,
                       get_image_shape, read_window, write_window,
                       create_empty_tiff)
from tqdm import tqdm
import gc

ROOT_PATH = dirname(__file__)

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
    for y in tqdm(range(0, height, BLOCK_SIZE//2)):

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
        
        gc.collect()
    
    # create the new image with the distance map
    save_lab = np.zeros(label_ref.shape)

    for obj in np.unique(label_ref[np.nonzero(label_ref)]):
        
        # normalize the distance map
        save_lab[label_ref==obj] = result[label_ref==obj]/np.max(result[label_ref==obj])
        
    return save_lab
    

def lazy_gaussian_distance_map(input_image_path:str, output_image_path:str, sigma = 5):
    
    BLOCK_SIZE = 5000

    img_metadata = get_image_metadata(input_image_path)
    img_shape = get_image_shape(input_image_path)

    temp_file = join(dirname(output_image_path), "temp_file.tiff")
    
    create_empty_tiff(
        filename = temp_file,
        dtype="float32",
        img_metadata=img_metadata,
        shape=img_shape
    )

    # Get the dimensions of the image
    height, width = img_shape[-2:]

    # Apply transformation to blocks with overlap
    for y in tqdm(range(0, height, BLOCK_SIZE//2), position=1):

        for x in tqdm(range(0, width, BLOCK_SIZE//2), position=2):
            
            # Define the boundaries of the block
            y_end = min(y + BLOCK_SIZE, height)
            x_end = min(x + BLOCK_SIZE, width)

            # Process the current block
            # block = label_ref[y:y_end, x:x_end]
            bbox = ((y, y_end), (x,x_end))

            input_img = read_window(input_image_path, bbox=bbox)

            # if all the values are 0, pass
            if np.all(input_img == 0):
                continue

            temp_img = read_window(temp_file, bbox=bbox)

            # convert into bool to improve label() func performance
            ref = (input_img > 0).astype("bool")

            # label the image as components
            label_ref = label(ref)
            
            # Apply euclidean distance transform
            depth_map = distance_transform_edt(label_ref)
            
            # Apply gaussian filter
            depth_map = gaussian_filter(depth_map, sigma)
            
            # Select the minimum non-zero value for each pixel
            depth_map = np.where(temp_img > 0,
                                   np.minimum(depth_map, temp_img),
                                   depth_map)
            
            # store block transformed
            write_window(
                temp_file, 
                depth_map,
                bbox
            )



    create_empty_tiff(
        filename = output_image_path,
        dtype="float32",
        img_metadata=img_metadata,
        shape=img_shape
    )


    # Apply transformation to blocks with overlap
    for y in tqdm(range(0, height, BLOCK_SIZE//2), position=1):

        for x in range(0, width, BLOCK_SIZE//2):
            
            # Define the boundaries of the block
            y_end = min(y + BLOCK_SIZE, height)
            x_end = min(x + BLOCK_SIZE, width)

            # Process the current block
            bbox = ((y, y_end), (x,x_end))

            temp_img = read_window(temp_file, bbox=bbox)
            
            label_final = label(temp_img > 0)

            norm_depth = np.zeros_like(temp_img)

            
            output_img = read_window(output_image_path, bbox=bbox)

            # iterate through labels to normalize
            for obj in np.unique(label_final[np.nonzero(label_final)]):
                
                # normalize the distance map
                norm_depth[label_final==obj] = temp_img[label_final==obj]/np.max(temp_img[label_final==obj])
            

            norm_depth = np.where(output_img > 0,
                                   np.minimum(norm_depth, output_img),
                                   norm_depth)
            
            # store block transformed
            write_window(
                output_image_path, 
                norm_depth,
                bbox
            )
    
    os.remove(temp_file)



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
    
    # train_input_path = args.train_segmentation_path
    train_input_path = join(ROOT_PATH, r"amazon_input_data\segmentation\train_set.tif")

    train_output_path = join(ROOT_PATH, "test_data", "train_distance_map.tif")
    
    check_folder(dirname(train_output_path))

    # generate_distance_map(train_input_path, train_output_path)
    lazy_gaussian_distance_map(train_input_path, train_output_path)


    print_sucess("Distance map generated successfully")