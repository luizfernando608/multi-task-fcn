import os
from os.path import isfile
from logging import getLogger
import numpy as np

from src.utils import array2raster, check_folder, get_image_metadata, read_yaml
logger = getLogger("__main__")

def pred2raster(current_iter_folder, args):
    
    logger.info(f"Computing the mean between the {len(args.overlap)} slices")

    output_folder = os.path.join(current_iter_folder, 'raster_prediction')
    check_folder(output_folder)

    prediction_file = os.path.join(
        output_folder,
        f'join_class_{np.sum(args.overlap)}.TIF'
        )
    
    prob_file = os.path.join(
        output_folder,
        f'join_prob_{np.sum(args.overlap)}.TIF')
    
    
    depth_file = os.path.joiget_image_metadatan(
        output_folder,
        f'depth_{np.sum(args.overlap)}.TIF')
               
    
    if not (isfile(prediction_file) and isfile(prob_file) and isfile(depth_file)):
    
        for ov in args.overlap:

            prediction_overlap_path = os.path.join(current_iter_folder, f'prediction_{ov}.npz')
            prediction_ov_data = np.load(prediction_overlap_path)
            
            try:
                prediction_test = np.add(prediction_test, prediction_ov_data["prob_map"])
                depth_test = np.add(depth_test, prediction_ov_data["depth_map"])

                
            except:
                prediction_test = np.load(prediction_ov_data["prob_map"])
                depth_test = np.load(prediction_ov_data["depth_map"])
            
            prediction_ov_data.close()
            os.remove(prediction_overlap_path)
        
        prediction_test/=len(args.overlap)
        depth_test/=len(args.overlap)

        # Get the metadata from segmentation file to save with the output with the same conf
        RASTER_PATH = args.train_segmentation_path
        image_metadata = get_image_metadata(RASTER_PATH)
        
        logger.info("Saving the raster prediction")
        
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
