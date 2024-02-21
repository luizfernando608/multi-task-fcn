import gc
import os
from logging import Logger
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from src.logger import create_logger
from src.model import build_model, load_weights
from src.multicropdataset import DatasetFromCoord
from src.utils import (add_padding_new, check_folder,
                       extract_patches_coord, get_device, get_image_metadata,
                       normalize, read_tiff, read_yaml, load_norm)

ROOT_PATH = os.path.dirname(__file__)
args = read_yaml(os.path.join(ROOT_PATH, "args.yaml"))


        
def define_test_loader(ortho_image:str, size_crops:int, overlap_rate:float)->Tuple:
    """Define the PyTorch loader for evaluation.\\
    This loader is different from the trainning loader.\\
    Here, the loader gets patches from the entire image map.\\
    On the other hand, the training loader just loads patches with some segmentation

    Parameters
    ----------
    ortho_image : str
        Path to the ortho_image. The image from remote sensing
    size_crops : int
        - The size of each patch    
    overlap_rate : float
        - The overlap rate between each patch

    Returns
    -------
    Tuple
        - image
            The normalized image from remote sensing
        - coords
            The coordinate of the center of each patch
        - stride
            The size of each step between each patch center
        - step_row
        - step_col
        - overlap
            The real overlap in pixels
    """

    
    
    image = load_norm(ortho_image)

    lab = np.ones(image.shape[1:])
    lab[np.sum(image, axis=0) == (11*image.shape[0]) ] = 0
    
    image, stride, step_row, step_col, overlap, _, _ = add_padding_new(image, size_crops, overlap_rate)
    
    coords = extract_patches_coord(
        img_gt = lab, 
        psize = size_crops,
        stride = stride, 
        step_row = step_row,
        step_col = step_col,
        overl = overlap_rate
    )

    return image, coords, stride, overlap


def predict_network(ortho_image_shape:Tuple, 
                    dataloader:torch.utils.data.DataLoader, 
                    model:nn.Module, 
                    batch_size:int, 
                    coords:np.ndarray, 
                    pred_prob:np.ndarray,
                    pred_depth:np.ndarray,
                    stride:int, 
                    overlap:int):
    """
    It runs the inference of the entire image map.\\
    Get depth values and the probability of each class

    Parameters
    ----------
    ortho_image_shape : Tuple
        The shape of the orthoimage
    dataloader : torch.utils.data.DataLoader
        Torch dataloader with all the patches from the image to be evaluated
    model : nn.Module
        The model with the weight of the current iteration
    batch_size : int
        The batch size at each prediction
    coords : np.ndarray
        The array with the coordinates of the patch centers
    pred_prob : np.ndarray
        An empty tensor to be filled with the probability values generated by the model
    pred_depth : np.ndarray
        An empty matrice to be filled with the depth values generated by the model
    stride : int
        The step between each patch
        The model use this to fill the pred_prob and pred_depth
    overlap : int
        The overlap between each patch in pixels

    Returns
    -------
    Tuple
        - pred_prob :  Probability of each class
        - The class with the highest probability value
        - pred_depth :  The depth_map generated by the model
    """
    DEVICE = get_device()
    model.eval()
    
    soft = nn.Softmax(dim=1).to(DEVICE)
    sig = nn.Sigmoid().to(DEVICE)
    
    st = stride//2
    ovr = overlap//2
    
    j = 0
    with torch.no_grad(): 
        for i, inputs in enumerate(tqdm(dataloader)):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = inputs.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            out_pred = model(input_batch) 
               
            out_batch = soft(out_pred['out'])
            out_batch = out_batch.permute(0,2,3,1)
                
            out_batch = out_batch.data.cpu().numpy()
            
            depth_out = sig(out_pred['aux']).data.cpu().numpy()
            
            c, x, y, cl = out_batch.shape

            coord_x = coords[j : j+batch_size,0]
            coord_y = coords[j : j+batch_size,1]

            # iterate through batches
            for b in range(c):
                pred_prob[
                    coord_x[b] - st : coord_x[b] + st + stride % 2,
                    coord_y[b] - st : coord_y[b] + st + stride % 2
                    ] = out_batch[ b , overlap//2 + overlap % 2 : x - ovr , overlap//2 + overlap % 2 : y - ovr ]

                pred_depth[
                    coord_x[b] - st : coord_x[b] + st + stride % 2,
                    coord_y[b] - st : coord_y[b] + st + stride % 2
                    ] = depth_out[b, 0, overlap//2 + overlap % 2 : x - ovr, overlap//2 + overlap % 2: y - ovr ]

            j += out_batch.shape[0] 
            
        
        row = ortho_image_shape[0]
        col = ortho_image_shape[1]
        
        pred_prob = pred_prob[overlap//2 + overlap % 2 :, overlap//2 + overlap % 2 :]
        pred_prob = pred_prob[:row,:col]
        
        pred_depth = pred_depth[overlap//2 + overlap % 2 :, overlap//2 + overlap % 2 :]
        pred_depth = pred_depth[:row,:col]
        

        return pred_prob, np.argmax(pred_prob,axis=-1), pred_depth


def evaluate_overlap(overlap:float,
                     current_iter_folder:str,
                     current_model_folder:str, 
                     ortho_image_shape:tuple,
                     logger:Logger, 
                     size_crops:int = args.size_crops, 
                     num_classes:int = args.nb_class,
                     ortho_image:str = args.ortho_image, 
                     batch_size:int = args.batch_size, 
                     workers:int = args.workers, 
                     checkpoint_file:str = args.checkpoint_file, 
                     arch:str = args.arch, 
                     filters:list = args.filters, 
                     is_pretrained:bool = args.is_pretrained):
    """This function runs an evaluation on the entire image.\\
    The image is divided into patches, that will be the inputs of the model.\\
    The overlap parameter sets overlap rate between each patch cut.

    Parameters
    ----------
    overlap : float
        Overlap rate between each patch
    current_iter_folder : str
        The path to the current iteration folder
    current_model_folder : str
        The path to the current model folder
    ortho_image_shape : tuple
        The shape the ortho image - Image from remote sensing
    logger : Logger
        The logger that tracks the model evaluation
    size_crops : int, optional
        The size of the patches, by default args.size_crops
    num_classes : int, optional
        Num of tree_types, by default args.nb_class
    ortho_image : str, optional
        The path to the image from remote sensing, by default args.ortho_image
    batch_size : int, optional
        The batch size of the stochastic trainnig, by default args.batch_size
    workers : int, optional
        Num of parallel workers, by default args.workers
    checkpoint_file : str, optional
        The filename of the checkpoint, by default args.checkpoint_file
    arch : str, optional
        Architecture name to build the model
        The architecture can be 'resunet' or 'deeplabv3_resnet50', by default args.arch
    filters : list, optional
        List of number of filters for each block of the model, if the model is resunet, by default args.filters
    is_pretrained : bool, optional
        If True, the model is loaded from pretrained weights available in pytorch hub
        If False, the model is started with random weights, by default args.is_pretrained
    """
    
    DEVICE = get_device()

    image, coords, stride, overlap_in_pixels = define_test_loader(ortho_image, 
                                                                    size_crops, 
                                                                    overlap,
                                                                    )

    test_dataset = DatasetFromCoord(
            image,
            labels = None,
            depth_img = None,
            coords = coords,
            psize = size_crops,
            evaluation = True,
        )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
    )

    logger.info("Building data done with {} patches loaded.".format(coords.shape[0]))
    

    model = build_model(
        image.shape, 
        args.nb_class,
        args.arch, 
        args.filters, 
        args.is_pretrained,
        psize = args.size_crops,
        dropout_rate = args.dropout_rate
    )


    last_checkpoint = os.path.join(current_model_folder, checkpoint_file)
    model = load_weights(model, last_checkpoint)
    logger.info("Model loaded from {}".format(last_checkpoint))

    # Load model to GPU
    model = model.to(DEVICE)

    cudnn.benchmark = True

    check_folder(os.path.join(current_iter_folder, 'prediction'))

    pred_prob = np.zeros(shape = (image.shape[1], image.shape[2], num_classes), dtype='float16')
    pred_depth = np.zeros(shape = (image.shape[1], image.shape[2]), dtype='float16')

    prob_map, pred_class, depth_map = predict_network(
        ortho_image_shape = ortho_image_shape,
        dataloader = test_loader,
        model = model,
        batch_size = batch_size,
        coords = coords,
        pred_prob = pred_prob,
        pred_depth = pred_depth,
        stride = stride,
        overlap = overlap_in_pixels,
    )

    gc.collect()

    prob_map_path = os.path.join(current_iter_folder, 'prediction', f'prob_map_{overlap}.npy')
    np.save(prob_map_path, prob_map)
    del prob_map
    gc.collect()
    logger.info("Probability map done and saved.")
    

    pred_class_path = os.path.join(current_iter_folder, 'prediction', f'pred_class_{overlap}.npy')
    np.save(pred_class_path, pred_class)
    del pred_class
    gc.collect()
    logger.info("Prediction done and saved.")
    
    
    depth_map_path = os.path.join(current_iter_folder, 'prediction', f'depth_map_{overlap}.npy')
    np.save(depth_map_path, depth_map)
    del depth_map
    gc.collect()
    logger.info("Depth map done and saved.")





def evaluate_iteration(current_iter_folder:str, args:dict):
    """Evaluate the entire image to predict the segmentation map.
    Evaluate for many overlap values and save the results in the current_iter_folder/prediction folder.

    Parameters
    ----------
    current_iter_folder : str
        Path to the current iteration folder.
    args : dict
        Dictionary of arguments.
    """

    logger = create_logger(os.path.join(current_iter_folder, "inference.log"))
    logger.info("============ Initialized Evaluation ============")

    ## show each args
    # logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    overlaps = args.overlap


    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    ortho_image_metadata = get_image_metadata(args.ortho_image)
    
    ortho_image_shape = (ortho_image_metadata["height"], ortho_image_metadata["width"])
    

    # Iterate over the different overlap values
    for overlap in overlaps:

        # Verify if the prediction is already done
        is_depth_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'depth_map_{overlap}.npy'))
        is_prob_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'prob_map_{overlap}.npy'))
        is_pred_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'pred_class_{overlap}.npy'))
        
        if is_depth_done and is_prob_done and is_pred_done:

            logger.info(f"Overlap {overlap} is already done. Skipping...")

            continue
        
        logger.info(f"Overlap {overlap} is not done. Starting...")
        evaluate_overlap(overlap, 
                         current_iter_folder, 
                         current_model_folder,
                         ortho_image_shape, 
                         logger)
        
        logger.info(f"Overlap {overlap} done.")

        gc.collect()
        torch.cuda.empty_cache()


##############################
#### E V A L U A T I O N #####
##############################
if __name__ == "__main__":
    ## arguments
    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = os.path.join(args.data_path, "iter_001")
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    evaluate_iteration(current_iter_folder, args)

    print("ok")
    


