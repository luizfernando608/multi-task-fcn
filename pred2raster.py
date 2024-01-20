import os
from os.path import isfile, join

import numpy as np
from tqdm import tqdm

from src.utils import array2raster, check_folder, fix_relative_paths,get_image_metadata, read_yaml, get_image_shape, read_window, write_window


def pred2raster(current_iter_folder, args):

    output_folder = join(current_iter_folder, 'raster_prediction')
    check_folder(output_folder)

    img_metadata = get_image_metadata(args.train_segmentation_path)
    img_shape = get_image_shape(args.train_segmentation_path)

    prediction_file = join(
        output_folder,
        f'join_class_itc{args.test_itc}_{np.sum(args.overlap)}.TIF'
    )
    
    prob_file = join(
        output_folder,
        f'join_prob_itc{args.test_itc}_{np.sum(args.overlap)}.TIF'
    )

    
    depth_file = join(
        output_folder,
        f'depth_itc{args.test_itc}_{np.sum(args.overlap)}.TIF'
    )
    
               
    
    if not (isfile(prediction_file) and isfile(prob_file) and isfile(depth_file)):
        
        # create empty tiff files
        array2raster(
            prediction_file,
            np.zeros(img_shape[1:],  dtype = "uint16"),
            img_metadata,
            dtype = "uint16"
        )

        array2raster(
            prob_file,
            np.zeros(img_shape[1:],  dtype = "float32"),
            img_metadata,
            dtype = "float32"
        )


        array2raster(
            depth_file,
            np.zeros(img_shape[1:],  dtype = "float32"),
            img_metadata,
            dtype = "float32"
        )

        PSIZE = 1000

        for row in tqdm(range(0, img_shape[-2], PSIZE)):
            for col in range(0, img_shape[-1], PSIZE):
                
                depth_list = []
                prob_list = []

                for ov in args.overlap:
                    
                    prob_path = join(current_iter_folder,'prediction',f'prob_map_itc{args.test_itc}_{ov}.tiff')
                    depth_path = join(current_iter_folder,'prediction',f'depth_map_itc{args.test_itc}_{ov}.tiff')

                    row_end = min([row + PSIZE, img_shape[-2]])

                    col_end = min([col + PSIZE, img_shape[-1]])

                    bbox = ((row, row_end), (col, col_end))
                    
                    # read windows and concat
                    prob_list.append(
                        read_window(
                            prob_path,
                            bbox,
                        )
                    )

                    depth_list.append(
                        read_window(
                            depth_path,
                            bbox,
                        )
                    )
                
                
                
                final_prob = np.mean(np.stack(prob_list, axis = 0), axis = 0)

                write_window(
                    prob_file,
                    np.amax(final_prob, axis = 0),
                    bbox
                )


                write_window(
                    prediction_file,
                    np.argmax(final_prob, axis = 0),
                    bbox
                )

                final_depth = np.mean(np.stack(depth_list, axis = 0), axis = 0)

                write_window(
                    depth_file,
                    final_depth,
                    bbox = bbox
                )
            
            
            # calculate the mean
            
            # write window with the mean
            


if __name__ == "__main__":

    args = read_yaml("args.yaml")
    
    fix_relative_paths(args)

    # external parameters
    current_iter_folder = join(args.data_path, "iter_001")

    current_iter = int(current_iter_folder.split("_")[-1])

    current_model_folder = join(current_iter_folder, args.model_dir)

    pred2raster(current_iter_folder, args)
