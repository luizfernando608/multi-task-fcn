import os
from os.path import isfile

import numpy as np

from src.utils import array2raster, check_folder, get_image_metadata, read_yaml


def pred2raster(current_iter_folder, args):

    output_folder = os.path.join(current_iter_folder, 'raster_prediction')
    check_folder(output_folder)

    prediction_file = os.path.join(
        output_folder,
        f'join_class_{np.sum(args.overlap)}.TIF'
        )
    
    prob_file = os.path.join(
        output_folder,
        f'join_prob_{np.sum(args.overlap)}.TIF')
    
    
    depth_file = os.path.join(
        output_folder,
        f'depth_{np.sum(args.overlap)}.TIF')
               
    
    if not (isfile(prediction_file) and isfile(prob_file) and isfile(depth_file)):
    
        for ov in args.overlap:
            try:
                prediction_path = os.path.join(current_iter_folder,'prediction',f'prob_map_{ov}.npy')
                prediction_test = np.add(prediction_test, np.load(prediction_path))
                
                depth_path = os.path.join(current_iter_folder,'prediction',f'depth_map_{ov}.npy')
                depth_test = np.add(depth_test,np.load(depth_path))
                
            except:
                prediction_path = os.path.join(current_iter_folder,'prediction',f'prob_map_{ov}.npy')
                prediction_test = np.load(prediction_path)
                
                depth_path = os.path.join(current_iter_folder,'prediction',f'depth_map_{ov}.npy')
                depth_test = np.load(depth_path)
                
        
        prediction_test/=3
        depth_test/=3

        # Get the metadata from segmentation file to save with the output with the same conf
        RASTER_PATH = args.train_segmentation_path
        image_metadata = get_image_metadata(RASTER_PATH)
        
        # Save the file
        array2raster(prediction_file, np.argmax(prediction_test, axis = -1), image_metadata, "uint8")    
        array2raster(prob_file, np.amax(prediction_test, axis = -1), image_metadata, "float32")  
        array2raster(depth_file, depth_test, image_metadata, "float32")   



if __name__ == "__main__":

    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_1"
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    pred2raster(current_iter_folder, args)
