#%%
import argparse
import math
import os
from logging import getLogger

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import glob
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skimage.morphology import dilation, disk

from tqdm import tqdm

from src.metrics import evaluate_metrics
from src.logger import create_logger
from src.utils import extract_patches_coord, add_padding_new, array2raster
from src.utils import (
    bool_flag,
    read_tiff,
    load_norm,
    check_folder,
    read_yaml
)

from src.multicropdataset import DatasetFromCoord
from src.resnet import ResUnet
from src.model import build_model, load_weights

args = read_yaml("args.yaml")


        
def define_test_loader(ortho_image, size_crops, overlap, test_itc, lab = None,):
    
    if not test_itc:
        image = read_tiff(ortho_image)
        lab = np.ones(image.shape[1:])
        lab[np.sum(image,axis=0)==(11*image.shape[0])] = 0
        del image

    image = load_norm(ortho_image)
    
    coords, _ = extract_patches_coord(lab, size_crops, overlap)
    image, stride, step_row, step_col, overlap, _, _ = add_padding_new(image, size_crops, overlap)

    return image, coords, stride, step_row, step_col, overlap


def predict_network(ortho_image_shape,dataloader, model, batch_size, coords, pred_prob, pred_depth,
                     stride, step_row, step_col, overlap):
    
    model.eval()
    
    soft = nn.Softmax(dim=1).cuda()
    sig = nn.Sigmoid().cuda()
    
    st = stride//2
    ovr = overlap//2
    
    j = 0
    with torch.no_grad(): 
        for i, inputs in enumerate(tqdm(dataloader)):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = inputs.cuda(non_blocking=True)
            out_pred = model(input_batch) 
               
            out_batch = soft(out_pred['out'])
            out_batch = out_batch.permute(0,2,3,1)
                
            out_batch = out_batch.data.cpu().numpy()
            
            depth_out = sig(out_pred['aux']).data.cpu().numpy()
            
            c, x, y, cl = out_batch.shape
            coord_x = coords[j:j+batch_size,0]
            coord_y = coords[j:j+batch_size,1]
            for b in range(c):
                pred_prob[coord_x[b]-st:coord_x[b]+st+stride%2,
                        coord_y[b]-st:coord_y[b]+st+stride%2] = out_batch[b,overlap//2:x-ovr+overlap%2,overlap//2:y-ovr+overlap%2]

                pred_depth[coord_x[b]-st:coord_x[b]+st+stride%2,
                        coord_y[b]-st:coord_y[b]+st+stride%2] = depth_out[b,0,overlap//2:x-ovr+overlap%2,overlap//2:y-ovr+overlap%2]

            j+=out_batch.shape[0] 
            
            
        # raster_src = gdal.Open(ortho_image)
        # row, col = raster_src.RasterYSize, raster_src.RasterXSize
        row = ortho_image_shape[0]
        col = ortho_image_shape[1]
        
        pred_prob = pred_prob[overlap//2:,overlap//2:]
        pred_prob = pred_prob[:row,:col]
        
        pred_depth = pred_depth[overlap//2:,overlap//2:]
        pred_depth = pred_depth[:row,:col]
        
        # np.save(os.path.join(args.model_dir,'prediction','prob_map_itc{}_{}'.format(args.test_itc,args.overlap)), pred_prob)
        # np.save(os.path.join(args.model_dir,'prediction','pred_class_itc{}_{}'.format(args.test_itc,args.overlap)), np.argmax(pred_prob,axis=-1))    
        # np.save(os.path.join(args.model_dir,'prediction','depth_map_itc{}_{}'.format(args.test_itc,args.overlap)), pred_depth)
        return pred_prob, np.argmax(pred_prob,axis=-1), pred_depth

#%%



def evaluate_overlap(overlap, ref, current_iter_folder,current_model_folder, ortho_image_shape,logger, size_crops=args.size_crops, num_classes=args.nb_class,
                     ortho_image=args.ortho_image, test_itc=args.test_itc, batch_size=args.batch_size, workers=args.workers, 
                     checkpoint_file=args.checkpoint_file, arch=args.arch, filters=args.filters, is_pretrained=args.is_pretrained):
                    

    image, coords, stride, step_row, step_col, overlap_in_pixels = define_test_loader(ortho_image, size_crops, overlap, test_itc, ref)

    pred_prob = np.zeros(shape = (image.shape[1],image.shape[2], num_classes), dtype='float16')
    pred_depth = np.zeros(shape = (image.shape[1],image.shape[2]), dtype='float16')

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
    model = build_model(image.shape, num_classes, arch, filters, is_pretrained)

    last_checkpoint = os.path.join(current_model_folder, checkpoint_file)
    model = load_weights(model, last_checkpoint, logger)

    # Load model to GPU
    model = model.cuda()

    cudnn.benchmark = True

    check_folder(os.path.join(current_iter_folder, 'prediction'))

    prob_map, pred_class, depth_map = predict_network(
        ortho_image_shape=ortho_image_shape,
        dataloader=test_loader,
        model=model,
        batch_size=batch_size,
        coords=coords,
        pred_prob=pred_prob,
        pred_depth=pred_depth,
        stride=stride,
        step_row=step_row,
        step_col=step_col,
        overlap=overlap_in_pixels,
    )
    prob_map_path = os.path.join(current_iter_folder, 'prediction', f'prob_map_itc{test_itc}_{overlap}.npy')
    np.save(prob_map_path, prob_map)
    del prob_map
    pred_class_path = os.path.join(current_iter_folder, 'prediction', f'pred_class_itc{test_itc}_{overlap}.npy')
    np.save(pred_class_path, pred_class)
    del pred_class    
    depth_map_path = os.path.join(current_iter_folder, 'prediction', f'depth_map_itc{test_itc}_{overlap}.npy')
    np.save(depth_map_path, depth_map)
    del depth_map




def evaluate_iteration(current_iter_folder, args):
    logger = create_logger(os.path.join(current_iter_folder, "inference.log"),rank=0)
    logger.info("============ Initialized Evaluation ============")
    ## show each args
    # logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    overlaps = args.overlap
    test_itc = args.test_itc

    segmentation_img_path = os.path.join(args.data_path, "segmentation", args.train_segmentation_file)

    current_model_folder = os.path.join(current_iter_folder, args.model_dir)

    temp_image = gdal.Open(args.ortho_image)
    ortho_image_shape = (temp_image.RasterYSize, temp_image.RasterXSize)
    del temp_image

    if test_itc:
        ref = read_tiff(segmentation_img_path)
    else:
        ref = None

    for overlap in overlaps:
        is_depth_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'depth_map_itc{test_itc}_{overlap}.npy'))
        is_prob_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'prob_map_itc{test_itc}_{overlap}.npy'))
        is_pred_done = os.path.exists(os.path.join(current_iter_folder, 'prediction', f'pred_class_itc{test_itc}_{overlap}.npy'))
        if is_depth_done and is_prob_done and is_pred_done:
            logger.info(f"Overlap {overlap} is already done. Skipping...")
            continue
        
        evaluate_overlap(overlap, ref, current_iter_folder,current_model_folder, ortho_image_shape, logger)
        gc.collect()
        torch.cuda.empty_cache()


##############################
#### E V A L U A T I O N #####
##############################
if __name__ == "__main__":
    ## arguments
    args = read_yaml("args.yaml")
    # external parameters
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_1"
    current_iter = int(current_iter_folder.split("_")[-1])
    current_model_folder = os.path.join(current_iter_folder, args.model_dir)


    evaluate_iteration(current_iter_folder, args)
    


