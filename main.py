#%%
import argparse
import math
import os

from os.path import dirname

import subprocess

from osgeo import gdal

import pandas as pd

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

from generate_distance_map import generate_distance_map

import matplotlib.pyplot as plt
plt.set_loglevel(level = 'info')

from src.logger import create_logger
from src.model import define_loader, build_model, load_weights, train, save_checkpoint
from src.multicropdataset import DatasetFromCoord
from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    read_tiff,
    check_folder,
    print_sucess,
    oversamp,
    read_yaml,
    array2raster
)

from evaluation import evaluate_iteration
from pred2raster import pred2raster
from sample_selection import get_new_segmentation_sample

import gc
gc.set_threshold(0)


def clear_ram_cache():
    """Execute some command in Unix kernel the free up garbage from the cache
    """
    clear_command = 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches'
    p = subprocess.Popen(clear_command, shell=True).wait()
    
    # alias freecachemem='sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null'
    clear_command_alias = "alias freecachemem='sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null'"
    p = subprocess.Popen(clear_command_alias, shell=True).wait()


def is_iter_0_done(data_path:str):
    """Verify if the distance map from the ground truth segmentation is done

    Parameters
    ----------
    data_path : str
        Root data path

    Returns
    -------
    bool
    """
    path_distance_map = os.path.join(data_path, "iter_000", "distance_map")
    is_test_map_done = os.path.exists(os.path.join(path_distance_map, "test_distance_map.tif"))
    is_train_map_done = os.path.exists(os.path.join(path_distance_map, "train_distance_map.tif"))
    
    if is_test_map_done and is_train_map_done:
        return True
    
    else:
        return False


def get_current_iter_folder(data_path, test_itc, overlap):
    """Get the current iteration folder
    This function verify which iteration folder isn't finished yet and return the path to it.

    Parameters
    ----------
    data_path : str
        Path to the data folder 
    test_itc : bool
        Parameter used for create the output file name
    overlap : float
        Parameter used for create the output file name

    Returns
    -------
    str
        The path to the current iteration folder
    """

    folders = pd.Series(os.listdir(data_path))

    folders = folders[folders.str.contains("iter_")]

    # num_folders = folders.str.replace("iter_", "").astype(int).sort_values(ascending=False)

    # iter_folders = ["iter_"+str(i) for i in num_folders.to_list()]

    iter_folders = folders.sort_values(ascending=False)


    for idx, iter_folder_name in enumerate(iter_folders):
        
        iter_path = os.path.join(data_path, iter_folder_name)

        prediction_path = os.path.join(iter_path, "raster_prediction")
        
        is_folder_generated = os.path.exists(prediction_path)
        is_depth_done = os.path.isfile(os.path.join(prediction_path, f"depth_itc{test_itc}_{np.sum(overlap)}.TIF"))
        is_pred_done = os.path.isfile(os.path.join(prediction_path, f"join_class_itc{test_itc}_{np.sum(overlap)}.TIF"))
        is_prob_done = os.path.isfile(os.path.join(prediction_path, f"join_prob_itc{test_itc}_{np.sum(overlap)}.TIF"))
        
        distance_map_path = os.path.join(iter_path, "distance_map")
        is_distance_map_done = os.path.isfile(os.path.join(distance_map_path, "selected_distance_map.tif"))

        if is_folder_generated and is_depth_done and is_pred_done and is_prob_done and is_distance_map_done:
            
            next_iter = int(iter_folder_name.split("_")[-1]) + 1
            next_iter_path = os.path.join(data_path, f"iter_{next_iter:03d}")
            
            check_folder(next_iter_path)

            return next_iter_path
    
    if is_iter_0_done(args.data_path):
        return os.path.join(data_path, f"iter_{1:03d}")
    

    iter_0_path = os.path.join(data_path, "iter_000")
    # create iter_0 folder if not exists
    check_folder(iter_0_path)

    return iter_0_path
    

def read_last_segmentation(current_iter_folder:str, train_segmentation_path:str)-> np.ndarray:
    """Read the segmentation labels from the last iteration.
    If is the first iteration, the function reads the ground_truth_segmentation
    If is not, the function reads the output from the last iteration in the folder `new_labels/`

    Parameters
    ----------
    current_iter_folder : str
        The folder of the current iteration
    train_segmentation_path : str
        The path to the ground truth segmentation

    Returns
    -------
    np.ndarray
        A image array with the segmentation set
    """

    
    current_iter = int(current_iter_folder.split("_")[-1])

    data_path = dirname(current_iter_folder)

    if current_iter==1:
        image_path = os.path.join(data_path, train_segmentation_path)
        

    else:
        image_path = os.path.join(data_path, f"iter_{current_iter-1:03d}", "new_labels", "selected_labels_set.tif")


    image = read_tiff(image_path)


    return image

    

def read_last_distance_map(current_iter_folder:str)->np.ndarray:
    """Read the last segmentation file with gaussian filter and distance map applied.
    If the current iter is 1, the distance map from the ground truth segmentation is loaded

    Parameters
    ----------
    current_iter_folder : str
        Current iteration folde

    Returns
    -------
    np.ndarray
        Image with the application of distance map.
    """
    current_iter = int(current_iter_folder.split("_")[-1])
    data_path = dirname(current_iter_folder)

    if current_iter == 1:
        distance_map_filename = "train_distance_map.tif"
    else:
        distance_map_filename = "selected_distance_map.tif"

    image_path = os.path.join(data_path, f"iter_{current_iter-1:03d}", "distance_map", distance_map_filename)
        
    image = read_tiff(image_path)
    return image


def get_learning_rate_schedule(train_loader: torch.utils.data.DataLoader, 
                               base_lr:float,
                               final_lr:float, 
                               epochs:int, 
                               warmup_epochs:int, 
                               start_warmup:float)->np.ndarray:
    """Get the learning rate schedule using cosine annealing with warmup
    
    This schedule start with a warmup learning rate and then decrease to the final learning rate

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Model train loader
    base_lr : float
        base learning rate
    final_lr : float
        final learning rate
    epochs : int
        number of total epochs to run
    warmup_epochs : int
        number of warmup epochs
    start_warmup : float
        initial warmup learning rate

    Returns
    -------
    np.array
        learning rate schedule
    """

    # define a linear distribuition from start_warmup to base_lr
    warmup_lr_schedule = np.linspace(start_warmup, base_lr, len(train_loader) * warmup_epochs)
    
    # iteration numbers
    iters = np.arange(len(train_loader) * (epochs - warmup_epochs))

    cosine_lr_schedule = []

    for t in iters:

        lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / (len(train_loader) * (epochs - warmup_epochs))))

        cosine_lr_schedule.append(lr)

    cosine_lr_schedule = np.array(cosine_lr_schedule)

    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    return lr_schedule



def train_epochs(last_checkpoint:str, 
                 start_epoch:str, 
                 num_epochs:int, 
                 best_val:float, 
                 train_loader:torch.utils.data.DataLoader, 
                 model:nn.Module, 
                 optimizer:torch.optim.Optimizer, 
                 lr_schedule:np.ndarray, 
                 rank:int, 
                 count_early:int, 
                 patience:int=5):
    """Train the model with the specified epochs numbers

    Parameters
    ----------
    last_checkpoint : str
        last checkpoint file path to load
    start_epoch : str
        Epoch num to start from
    num_epochs : int
        Total number fo epochs to execute
    best_val : float
        Best value got in this iteration
    train_loader : torch.utils.data.DataLoader
        The train dataload
    model : nn.Module
        Pytorch model
    optimizer : torch.optim.optimizer
        Pytorch optimizer
    lr_schedule : np.ndarray
        Learning rate schedule to use at each iteration
    rank : int
        
    count_early : int
        The counting how many epochs we didnt have improving in the loss
    patience : int, optional
        The limit of the count early variable, by default 5
    """

    # Create figures folder to save training figures every epoch
    figures_path = os.path.join(os.path.dirname(last_checkpoint), 'figures')
    check_folder(figures_path)

    for epoch in tqdm(range(start_epoch, num_epochs)):

        if rank == 0:
            if count_early == patience:
                logger.info("============ Early Stop at epoch %i ... ============" % epoch)
                break
        
        np.random.shuffle(train_loader.dataset.coord)

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train the network
        scores_tr = train(train_loader, model, optimizer, epoch, lr_schedule, figures_path, logger)
        
        gc.collect()

        training_stats.update(scores_tr)
        
        print_sucess("scores_tr: {}".format(scores_tr[1]))

        is_best = (best_val - scores_tr[1] ) > 0.001

        # save checkpoints
        if rank == 0:
            if is_best: 
                logger.info("============ Saving best models at epoch %i ... ============" % epoch)
                best_val = scores_tr[1]
                save_checkpoint(last_checkpoint, model, optimizer, epoch+1, best_val, count_early)                 
            else:
                count_early+=1

            
    print_sucess("Training done !")



def train_iteration(current_iter_folder:str, args:dict):
    """Train the model in the current iteration.
    Load the output and the model trained from the last iteration, and train the model again

    Parameters
    ----------
    current_iter_folder : str
        The current iteration folder
    args : dict
        The dict with the parameters for the model.
        The parameters are defined in the args.yaml file
    """

    

    current_iter = int(current_iter_folder.split("_")[-1])

    logger = create_logger(os.path.join(current_iter_folder, "train.log"), rank=0)
    logger.info("============ Initialized Training ============")
    
    ######### Define Loader ############
    raster_train = read_last_segmentation(current_iter_folder, args.train_segmentation_path)
    depth_img = read_last_distance_map(current_iter_folder)

    image, coords_train, raster_train, labs_coords_train = define_loader(args.ortho_image, 
                                                                         raster_train, 
                                                                         args.size_crops)

    ######## do oversampling in minor classes
    coords_train = oversamp(coords_train, labs_coords_train, under=False)

    if args.samples > coords_train.shape[0]:
        args.samples = None

    # build data for training
    train_dataset = DatasetFromCoord(
        image,
        raster_train,
        depth_img,
        coords_train,
        args.size_crops,
        args.samples,
        augment = args.augment
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )


    logger.info("Building data done with {} images loaded.".format(len(train_loader)))

    
    model = build_model(
        image.shape, 
        args.nb_class,  
        args.arch, 
        args.filters, 
        args.is_pretrained)


    ########## LOAD MODEL WEIGHTS FROM THE LAST CHECKPOINT ##########
    last_checkpoint = os.path.join(current_model_folder, args.checkpoint_file)
    
    loaded_from_last_checkpoint = False
    
    # If the weights from the current iteration, doenst exist. 
    # The weigths from the last one is loaded
    if not os.path.isfile(last_checkpoint):
        if current_iter > 1:
            last_checkpoint = os.path.join(args.data_path, f"iter_{current_iter-1:03d}", args.checkpoint_file)
            loaded_from_last_checkpoint = True
            print_sucess("Loaded_from_last_checkpoint")
        
        elif current_iter == 1:
            # The model will start with random weights
            pass


    model = load_weights(model, last_checkpoint, logger)
    ###################################################################
    
    # Load model to GPU
    model = model.cuda()


    logger.info("Building model done.")

    
    ###### BULD OPTMIZER #######
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # define how the learning rate will be changed in the training process.
    lr_schedule = get_learning_rate_schedule(
        train_loader, 
        args.base_lr, 
        args.final_lr, 
        args.epochs, 
        args.warmup_epochs, 
        args.start_warmup
    )
    logger.info("Building optimizer done.")


    #### LOAD METRICS FROM THE LAST CHECKPOINT ####
    to_restore = {"epoch": 0, "best_val":(100.), "count_early": 0, "is_iter_finished":False}
    restart_from_checkpoint(
        last_checkpoint,
        run_variables = to_restore,
        state_dict = model,
        optimizer = optimizer,
        logger = logger
    )


    # If the metrics are from the model from the last iteration, 
    # the model reset the metrics
    if loaded_from_last_checkpoint:
        to_restore["epoch"] = 0
        to_restore["best_val"] = 100.
        to_restore["count_early"] = 0
        to_restore["is_iter_finished"] = False
    

    ######## TRAIN MODEL #########
    current_checkpoint = os.path.join(current_model_folder, args.checkpoint_file)
    
    cudnn.benchmark = True
    
    gc.collect()

    # If the model isnt finished yet, train!
    if not to_restore["is_iter_finished"]:
        train_epochs(current_checkpoint, to_restore["epoch"], args.epochs, to_restore["best_val"] , train_loader, model, optimizer, lr_schedule, args.rank, to_restore["count_early"])

    gc.collect()


    #### CHANGE MODEL STATUS TO FINISHED ####
    # load models weights again to change status to is_iter_finished=True
    model = load_weights(model, current_checkpoint, logger)

    to_restore = {"epoch": 0, "count_early": 0, "is_iter_finished":False, "best_val":(100.)}
    restart_from_checkpoint(
        last_checkpoint,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        logger=logger
    )
    
    
    save_checkpoint(current_checkpoint, model, optimizer, to_restore["epoch"], to_restore["best_val"], to_restore["count_early"], is_iter_finished=True)
    

    # FREE UP MEMORY
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    gc.collect()


#############
### SETUP ###
#############


args = read_yaml("args.yaml")


##### LOOP #####

# Set random seed
fix_random_seeds(args.seed[0])

current_iter = 0

while current_iter < 10:
    clear_ram_cache()
    # get current iteration folder
    current_iter_folder = get_current_iter_folder(args.data_path, args.test_itc, args.overlap)
    current_iter = int(current_iter_folder.split("_")[-1])
    print("Current iteration folder: ", current_iter_folder)
    
    if current_iter == 0:
        
        test_segmentation_path = os.path.join(args.data_path, args.test_segmentation_path)
        test_distance_map_output = os.path.join(current_iter_folder, "distance_map", "test_distance_map.tif")
        
        check_folder(dirname(test_distance_map_output))

        generate_distance_map(test_segmentation_path, test_distance_map_output)

        train_segmentation_path = os.path.join(args.data_path, args.train_segmentation_path)
        train_distance_map = os.path.join(current_iter_folder, "distance_map", "train_distance_map.tif")

        check_folder(dirname(train_distance_map))

        generate_distance_map(train_segmentation_path, train_distance_map)
        continue

    # Get current model folder
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)
    check_folder(current_model_folder)

    logger, training_stats = initialize_exp(args, "epoch", "loss")

    train_iteration(current_iter_folder, args)

    evaluate_iteration(current_iter_folder, args)

    pred2raster(current_iter_folder, args)

    #####################################################
    ######### GENERATE LABELS FOR NEXT ITERATION #########
    new_pred_file = os.path.join(current_iter_folder, "raster_prediction", f'join_class_itc{args.test_itc}_{np.sum(args.overlap)}.TIF')
    new_pred_map = read_tiff(new_pred_file)

    new_prob_file = os.path.join(current_iter_folder, "raster_prediction", f'join_prob_itc{args.test_itc}_{np.sum(args.overlap)}.TIF')
    new_prob_map = read_tiff(new_prob_file)

    if current_iter == 1:
        old_pred_file = os.path.join(args.data_path, args.train_segmentation_path)

    else:
        old_pred_file = os.path.join(args.data_path, f"iter_{current_iter-1:03d}", "new_labels", f'selected_labels_set.tif')

    old_pred_map = read_tiff(old_pred_file)

    ground_truth_segmentation = read_tiff(os.path.join(args.data_path, args.train_segmentation_path))

    all_labels_set, selected_labels_set = get_new_segmentation_sample(
        ground_truth_map = ground_truth_segmentation,
        old_pred_map = old_pred_map, 
        new_pred_map = new_pred_map, 
        new_prob_map = new_prob_map, 
        data_path = args.data_path,
    )


    raster_src = gdal.Open(old_pred_file)

    all_labels_path = os.path.join(current_iter_folder, "new_labels", f'all_labels_set.tif')
    
    check_folder(os.path.dirname(all_labels_path))
    array2raster(all_labels_path, raster_src, all_labels_set, "Byte")


    selected_labels_path = os.path.join(current_iter_folder, "new_labels", f'selected_labels_set.tif')

    check_folder(os.path.dirname(selected_labels_path))
    array2raster(selected_labels_path, raster_src, selected_labels_set, "Byte")
    
    
    #######################################
    ######## GENERATE DISTANCE MAP ########
    all_labels_distance_map_path = os.path.join(current_iter_folder, "distance_map", f'all_labels_distance_map.tif')
    check_folder(os.path.dirname(all_labels_distance_map_path))
    generate_distance_map(all_labels_path, all_labels_distance_map_path)
    

    selected_distance_map_path = os.path.join(current_iter_folder, "distance_map", f'selected_distance_map.tif')
    check_folder(os.path.dirname(selected_distance_map_path))
    generate_distance_map(selected_labels_path, selected_distance_map_path)
    
    print_sucess("Distance map generated")
 


