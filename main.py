#%%
import argparse
import math
import os
from logging import getLogger
from logging import Logger

import shutil

import pandas as pd

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

from generate_distance_map import generate_train_test_map, is_generated

from src.logger import create_logger
from src.metrics import evaluate_metrics
from src.model import define_loader, build_model, load_weights, train, save_checkpoint
from src.multicropdataset import DatasetFromCoord
from src.resnet import ResUnet
from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    read_tiff,
    load_norm,
    check_folder,
    plot_figures,
    print_sucess,
    oversamp,
    read_yaml
)
from evaluation import evaluate_iteration
from pred2raster import pred2raster
def first_output_exists(data_path, test_itc, overlap):
    output_path = os.path.join(data_path, "iter_1", "raster_prediction")
    
    depth_path = os.path.join(output_path, f"depth_itc{test_itc}_{np.sum(overlap)}.TIF")
    is_depth_done = os.path.isfile(depth_path)

    prob_path = os.path.join(output_path, f"join_prob_itc{test_itc}_{np.sum(overlap)}.TIF")
    is_prob_done = os.path.isfile(prob_path)

    class_path = os.path.join(output_path, f"join_class_itc{test_itc}_{np.sum(overlap)}.TIF")
    is_class_done = os.path.isfile(class_path)

    return is_depth_done and is_prob_done and is_class_done

def is_first_iter(data_path, test_itc, overlap):
    """Check if it is the first iteration of the training process

    Returns
    -------
    bool
        True if it is the first iteration, False otherwise
    """
    return "iter_" not in os.listdir(data_path) and not first_output_exists(data_path, test_itc, overlap)


    
def get_iter_folder(data_path, test_itc, overlap):
    if is_first_iter(data_path, test_itc, overlap):
        print("First iteration")
        return os.path.join(data_path, "iter_1")
    else:
        paths = pd.Series(os.listdir(data_path))
        paths = paths[paths.str.contains("iter")].sort_values()
        last_iter_folder = paths.iloc[-1]
        return os.path.join(data_path, last_iter_folder)


def read_last_segmentation(current_iter_folder, data_path, train_segmentation_file):
    current_iter = int(current_iter_folder.split("_")[-1])
    if current_iter==1:
        image_path = os.path.join(data_path, "segmentation", train_segmentation_file)
        image = read_tiff(image_path)
        return image
    else:
        print(current_iter_folder)
        raise("Implementar leitura da próxima iteração")
    

def read_last_distance_map(current_iter_folder, data_path):
    current_iter = int(current_iter_folder.split("_")[-1])
    if current_iter == 1 :
        image_path = os.path.join(data_path, "before_iter", "train_distance_map.tif")
        image = read_tiff(image_path)
        return image
    else:
        print(current_iter_folder)
        raise("Implementar leitura da próxima iteração")



def get_learning_rate_schedule(train_loader: torch.utils.data.DataLoader,base_lr:float,final_lr:float, epochs:int, warmup_epochs:int, start_warmup:float)->np.array:
    """Get the learning rate schedule using cosine annealing with warmup

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
    # define learning rate schedule
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


def train_epochs(last_checkpoint, start_epoch, num_epochs, best_val, train_loader, model, optimizer, lr_schedule, rank, count_early, patience:int=20):
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
        
        training_stats.update(scores_tr)
        
        is_best = scores_tr[1] <= best_val

        # save checkpoints
        if rank == 0:
            if is_best:
                logger.info("============ Saving best models at epoch %i ... ============" % epoch)
                best_val = scores_tr[1]
                save_checkpoint(last_checkpoint, model, optimizer, epoch, best_val)
            else:
                count_early+=1
                
            
        print_sucess(logger)
        break
    
    



def train_iteration(current_iter_folder, args):
    ######### Define Loader ############
    raster_train = read_last_segmentation(current_iter_folder, args.data_path, args.train_segmentation_file)
    depth_img = read_last_distance_map(current_iter_folder, args.data_path)

    image, coords_train, raster_train, labs_coords_train = define_loader(args.ortho_image, raster_train, args.size_crops)    

    ######## do oversampling in minor classes
    coords_train = oversamp(coords_train, labs_coords_train,under=False)

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

    num_classes = len(np.unique(labs_coords_train))

    model = build_model(image, num_classes,  args.arch, args.filters, args.is_pretrained)

    last_checkpoint = os.path.join(current_model_folder, args.checkpoint_file)
    model = load_weights(model, last_checkpoint, logger)

    # Load model to GPU
    model = model.cuda()


    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    lr_schedule = get_learning_rate_schedule(
        train_loader, 
        args.base_lr, 
        args.final_lr, 
        args.epochs, 
        args.warmup_epochs, 
        args.start_warmup
    )
    logger.info("Building optimizer done.")


    to_restore = {"epoch": 0, "best_acc":(0.), "count_early":0}
    restart_from_checkpoint(
        last_checkpoint,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]
    best_val = to_restore["best_acc"]
    count_early = to_restore["count_early"]

    cudnn.benchmark = True
    train_epochs(last_checkpoint, start_epoch, args.epochs, best_val , train_loader, model, optimizer, lr_schedule, args.rank, count_early)

    del model

    with torch.no_grad():
        torch.cuda.empty_cache()

#############
### SETUP ###
#############

logger = getLogger('swav_model')
args = read_yaml("args.yaml")


##### LOOP #####

# Generate distance map if necessary
generate_train_test_map()
# Set random seed
fix_random_seeds(args.seed[0])


# get current iteration folder
current_iter_folder = get_iter_folder(args.data_path, args.test_itc, args.overlap)
current_iter = int(current_iter_folder.split("_")[-1])

# Create model folder for checkpoint
current_model_folder = os.path.join(current_iter_folder, args.model_dir)
check_folder(current_model_folder)


logger, training_stats = initialize_exp(args, "epoch", "loss")

train_iteration(current_iter_folder, args)

evaluate_iteration(current_iter_folder, args)

pred2raster(current_iter_folder, args)

# load and save the best model for the next iteration
next_iter_folder = os.path.join(args.data_path, "iter_"+str(current_iter+1))
check_folder(next_iter_folder)

# copy from  current_iter_folder to next_iter_folder



# save_checkpoint(
#     last_checkpoint_path = os.path.join(next_iter_folder, args.checkpoint_file), 
#     model = model, 
#     optimizer=optimizer, 
#     epoch=0 )





#%%%
