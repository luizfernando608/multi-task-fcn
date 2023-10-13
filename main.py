#%%
import argparse
import math
import os

from os.path import dirname, join, exists, isfile

import subprocess

import shutil

import pandas as pd

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

from generate_distance_map import generate_distance_map

from src.metrics import evaluate_metrics, evaluate_component_metrics
from src.utils import get_device, get_image_metadata, save_yaml

import matplotlib.pyplot as plt

import yaml



plt.set_loglevel(level = 'info')

from src.logger import create_logger
from src.model import define_loader, build_model, load_weights, train, save_checkpoint
from src.multicropdataset import DatasetFromCoord
from src.utils import (
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


def delete_useless_files(current_iter_folder:str):
    
    folder_to_remove = join(current_iter_folder,"prediction")

    if exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)
    
    else:
        raise FileExistsError("The path doesnt exist")




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
    path_distance_map = join(data_path, "iter_000", "distance_map")
    is_test_map_done = exists(join(path_distance_map, "test_distance_map.tif"))
    is_train_map_done = exists(join(path_distance_map, "train_distance_map.tif"))
    
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
        
        iter_path = join(data_path, iter_folder_name)

        prediction_path = join(iter_path, "raster_prediction")
        
        is_folder_generated = exists(prediction_path)
        is_depth_done = isfile(join(prediction_path, f"depth_itc{test_itc}_{np.sum(overlap)}.TIF"))
        is_pred_done = isfile(join(prediction_path, f"join_class_itc{test_itc}_{np.sum(overlap)}.TIF"))
        is_prob_done = isfile(join(prediction_path, f"join_prob_itc{test_itc}_{np.sum(overlap)}.TIF"))
        
        distance_map_path = join(iter_path, "distance_map")
        is_distance_map_done = isfile(join(distance_map_path, "selected_distance_map.tif"))

        if is_folder_generated and is_depth_done and is_pred_done and is_prob_done and is_distance_map_done:
            
            next_iter = int(iter_folder_name.split("_")[-1]) + 1
            next_iter_path = join(data_path, f"iter_{next_iter:03d}")
            
            check_folder(next_iter_path)

            return next_iter_path
    
    if is_iter_0_done(args.data_path):
        return join(data_path, f"iter_{1:03d}")
    

    iter_0_path = join(data_path, "iter_000")
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

    if current_iter == 1:
        image_path = train_segmentation_path
        

    else:
        image_path = join(data_path, f"iter_{current_iter-1:03d}", "new_labels", "selected_labels_set.tif")


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

    image_path = join(data_path, f"iter_{current_iter-1:03d}", "distance_map", distance_map_filename)
        
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
                 logger,
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
    
    device = get_device()
    # Create figures folder to save training figures every epoch
    figures_path = join(dirname(last_checkpoint), 'figures')
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

        # training_stats.update(scores_tr)
        
        print_sucess("scores_tr: {}".format(scores_tr[1]))

        is_best = (best_val - scores_tr[1] ) > 0.001

        # save checkpoints
        if rank == 0:
            if is_best: 
                logger.info("============ Saving best models at epoch %i ... ============" % epoch)
                
                best_val = scores_tr[1]
                
                save_checkpoint(last_checkpoint, model, optimizer, epoch+1, best_val, count_early)
                
                count_early = 0


            else:
                count_early += 1

            
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
    DEVICE = get_device()

    current_model_folder = join(current_iter_folder, args.model_dir)

    current_iter = int(current_iter_folder.split("_")[-1])

    logger = create_logger(join(current_iter_folder, "train.log"), rank=0)
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
    last_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    loaded_from_last_iteration = False
    
    # If the weights from the current iteration, doenst exist. 
    # The weigths from the last one is loaded
    if not isfile(last_checkpoint):
        if current_iter > 1:
            
            last_checkpoint = join(args.data_path, f"iter_{current_iter-1:03d}", args.model_dir, args.checkpoint_file)
            
            loaded_from_last_iteration = True

            print_sucess("Loaded_from_last_checkpoint")
        

        elif current_iter == 1:
            # The model will start with random weights
            pass


    model = load_weights(model, last_checkpoint, logger)
    ###################################################################
    
    # Load model to GPU
    model = model.to(DEVICE)


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
    if loaded_from_last_iteration:
        to_restore["epoch"] = 0
        to_restore["best_val"] = 100.
        to_restore["count_early"] = 0
        to_restore["is_iter_finished"] = False
    

    ######## TRAIN MODEL #########
    current_checkpoint = join(current_model_folder, args.checkpoint_file)
    
    cudnn.benchmark = True
    
    gc.collect()

    # If the model isnt finished yet, train!
    if not to_restore["is_iter_finished"]:
        train_epochs(current_checkpoint, 
                     to_restore["epoch"], 
                     args.epochs, 
                     to_restore["best_val"], 
                     train_loader, 
                     model, 
                     optimizer, 
                     lr_schedule, 
                     args.rank, 
                     to_restore["count_early"],
                     logger=logger)
    gc.collect()


    #### CHANGE MODEL STATUS TO FINISHED ####
    # load models weights again to change status to is_iter_finished=True
    model = load_weights(model, current_checkpoint, logger)

    to_restore = {"epoch": 0, "count_early": 0, "is_iter_finished":False, "best_val":(100.)}
    restart_from_checkpoint(
        current_checkpoint,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        logger=logger
    )
    
    
    save_checkpoint(current_checkpoint, 
                    model, 
                    optimizer, 
                    to_restore["epoch"], 
                    to_restore["best_val"], 
                    to_restore["count_early"], 
                    is_iter_finished=True)
    

    # FREE UP MEMORY
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    gc.collect()



def compile_metrics(current_iter_folder, args):
    # read test segmentation 
    # f'join_class_itc{args.test_itc}_{np.sum(args.overlap)}.TIF'
    DATA_PATH = dirname(current_iter_folder)

    GROUND_TRUTH_TEST_PATH = args.test_segmentation_path
    ground_truth_test = read_tiff(GROUND_TRUTH_TEST_PATH)

    GROUND_TRUTH_TRAIN_PATH = args.train_segmentation_path
    ground_truth_train = read_tiff(GROUND_TRUTH_TRAIN_PATH)

    PRED_PATH = join(current_iter_folder, "raster_prediction", f"join_class_itc{args.test_itc}_{np.sum(args.overlap)}.TIF")
    predicted_seg = read_tiff(PRED_PATH)

    ### Save test metrics ###
    metrics_test = evaluate_metrics(predicted_seg, ground_truth_test)

    save_yaml(metrics_test, join(current_iter_folder,'test_metrics.yaml'))    

    
    ### Save train metrics ###
    metrics_train = evaluate_metrics(predicted_seg, ground_truth_train, args.nb_class)
            
    save_yaml(metrics_train, join(current_iter_folder,'train_metrics.yaml'))
    

    ### Save test component metrics ###
    HIGH_PROB_COMPONENTS_PATH = join(current_iter_folder,'all_labels_test_metrics.yaml')
    
    all_labels = read_tiff(join(current_iter_folder, "new_labels", "all_labels_set.tif"))

    all_labels_metrics = evaluate_component_metrics(ground_truth_test, all_labels, 14)

    save_yaml(all_labels_metrics, HIGH_PROB_COMPONENTS_PATH)

    



#############
### SETUP ###
#############

ROOT_PATH = dirname(__file__)
args = read_yaml(join(ROOT_PATH, "args.yaml"))

# create output path
check_folder(args.data_path)

##### LOOP #####

# Set random seed
fix_random_seeds(args.seed[0])

while True:  
    print_sucess("Working ON:")
    print_sucess(get_device()) 
    
    clear_ram_cache()
    # get current iteration folder
    current_iter_folder = get_current_iter_folder(args.data_path, args.test_itc, args.overlap)
    current_iter = int(current_iter_folder.split("_")[-1])

    if current_iter > 20:
        break
    
    print("Current iteration folder: ", current_iter_folder)
    
    # if the iteration 0 applies distance map to ground truth segmentation
    if current_iter == 0:
        
        TEST_SEGMENTATION_PATH = args.test_segmentation_path
        test_distance_map_output = join(current_iter_folder, "distance_map", "test_distance_map.tif")
        
        check_folder(dirname(test_distance_map_output))

        generate_distance_map(TEST_SEGMENTATION_PATH, test_distance_map_output)

        TRAIN_SEGMENTATION_PATH  = args.train_segmentation_path
        train_distance_map = join(current_iter_folder, "distance_map", "train_distance_map.tif")

        check_folder(dirname(train_distance_map))

        generate_distance_map(TRAIN_SEGMENTATION_PATH, train_distance_map)
        continue
    
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    # Get current model folder
    current_model_folder = join(current_iter_folder, args.model_dir)
    check_folder(current_model_folder)

    # logger, training_stats = initialize_exp(current_iter_folder, args, "epoch", "loss")

    train_iteration(current_iter_folder, args)

    evaluate_iteration(current_iter_folder, args)

    pred2raster(current_iter_folder, args)

    

    #####################################################
    ######### GENERATE LABELS FOR NEXT ITERATION #########
    NEW_PRED_FILE = join(current_iter_folder, "raster_prediction", f'join_class_itc{args.test_itc}_{np.sum(args.overlap)}.TIF')
    new_pred_map = read_tiff(NEW_PRED_FILE)

    NEW_PROB_FILE = join(current_iter_folder, "raster_prediction", f'join_prob_itc{args.test_itc}_{np.sum(args.overlap)}.TIF')
    new_prob_map = read_tiff(NEW_PROB_FILE)

    NEW_DEPTH_FILE = join(current_iter_folder, "raster_prediction", f'depth_itc{args.test_itc}_{np.sum(args.overlap)}.TIF')
    new_depth_map = read_tiff(NEW_DEPTH_FILE)

    if current_iter == 1:
        OLD_PRED_FILE = args.train_segmentation_path

    else:
        OLD_PRED_FILE = join(args.data_path, f"iter_{current_iter-1:03d}", "new_labels", f'selected_labels_set.tif')

    old_pred_map = read_tiff(OLD_PRED_FILE)

    ground_truth_segmentation = read_tiff(args.train_segmentation_path)

    all_labels_set, selected_labels_set = get_new_segmentation_sample(
        ground_truth_map = ground_truth_segmentation,
        old_pred_map = old_pred_map, 
        new_pred_map = new_pred_map, 
        new_prob_map = new_prob_map, 
        new_depth_map = new_depth_map
    )

    # image metadata to save array2raster
    image_metadata = get_image_metadata(OLD_PRED_FILE)

    ALL_LABELS_PATH = join(current_iter_folder, "new_labels", f'all_labels_set.tif')
    
    check_folder(dirname(ALL_LABELS_PATH))
    array2raster(ALL_LABELS_PATH, all_labels_set, image_metadata, "Byte")


    SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", f'selected_labels_set.tif')

    check_folder(dirname(SELECTED_LABELS_PATH))
    array2raster(SELECTED_LABELS_PATH, selected_labels_set, image_metadata, "Byte")
    
    
    #######################################
    ######## GENERATE DISTANCE MAP ########
    ALL_LABELS_DISTANCE_MAP_PATH = join(current_iter_folder, "distance_map", f'all_labels_distance_map.tif')
    check_folder(dirname(ALL_LABELS_DISTANCE_MAP_PATH))
    generate_distance_map(ALL_LABELS_PATH, ALL_LABELS_DISTANCE_MAP_PATH)
    

    SELECTED_LABELS_DISTANCE_MAP_PATH  = join(current_iter_folder, "distance_map", f'selected_distance_map.tif')
    check_folder(dirname(SELECTED_LABELS_DISTANCE_MAP_PATH))
    generate_distance_map(SELECTED_LABELS_PATH, SELECTED_LABELS_DISTANCE_MAP_PATH)
    

    #############################################

    delete_useless_files(current_iter_folder = current_iter_folder)
    
    compile_metrics(current_iter_folder, args)

    print_sucess("Distance map generated")
 


