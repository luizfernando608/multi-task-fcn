import argparse
import math
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.metrics import evaluate_metrics
import glob
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skimage.morphology import dilation, disk

from tqdm import tqdm

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



parser = argparse.ArgumentParser(description="Inference of CNN-patch")

#########################
#### data parameters ####
#########################
# parser.add_argument('--orto_img', type=str, default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/NNDiffusePanSharpening_REFLECTANCE_TIFF_NOCLOUDS_8bits.tif',
#                         help="Path containing the raster image")

# parser.add_argument('--orto_img', type=str, default='/home/luiz/multi-task-fcn/Data/orthoimages/new_ortoA1_25tiff.tif',
#                         help="Path containing the raster image")
# parser.add_argument('--ref_path',type=str, default=None, 
#                     help="Path containing the refrence and mask data for training")
# parser.add_argument("--overlap", type=float, default=[0.1,0.3,0.5], 
#                     help="samples per epoch")
# parser.add_argument("--size_crops", type=int, default=256, 
#                     help="True for training 4 disjoint regions")
# parser.add_argument("--nb_class", type=int, default=8, 
#                     help="samples per epoch")
# parser.add_argument("--test_itc", type=bool, default=False, 
#                     help="True for predicting only the test ITCs")

#########################
#### model parameters ###
#########################
parser.add_argument("--patch_wise", default=False, type=bool, 
                    help="True for patch wise classification")
parser.add_argument("--arch", default="deeplabv3_resnet50", type=str, 
                    help="convnet architecture --> 'resnet18','resnet34','resnet50','resnet101', 'deeplabv3_resnet50'")
parser.add_argument("--load_model_from_path", default=True, type=bool, 
                    help="True for load model from path")


##########################
#### others parameters ###
##########################
parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default="./exp_deeplab_v4",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


#########################
#### model parameters ###
#########################
# Usar o último modelo treinado
parser.add_argument("--pretrained", default="checkpoint.pth.tar", type=str, 
                    help="path to pretrained weights")


#########################
#### optim parameters ###
#########################
parser.add_argument("--batch_size", default=8, type=int,
                    help="batch size ")


def main(overlap):
    global args
    # args = parser.parse_args()
    args = read_yaml("args.yaml")
    args.pretrained_file_checkpoint = os.path.join(args.model_dir, args.pretrained_file_checkpoint)
    
    args.overlap = overlap

    ######## get data #####

    
    # create a logger
    logger = create_logger(os.path.join(args.model_dir, "inference.log"),rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    if args.test_itc:
        ref = read_tiff(args.ref_path_evaluation)
        image, coords, stride, step_row, step_col, overlap  = define_loader(args.orto_img, ref)

    else:
        ref = None
        image, coords, stride, step_row, step_col, overlap = define_loader(args.ortho_image, ref)
    

    pred_prob = np.zeros(shape = (image.shape[1],image.shape[2],args.nb_class), dtype='float16')
    pred_depth = np.zeros(shape = (image.shape[1],image.shape[2]), dtype='float16')

    # build data for test set
    test_dataset = DatasetFromCoord(
        image,
        labels = None,
        depth_img = None,
        coords = coords,
        psize = args.size_crops,
        evaluation = True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
        
    logger.info("Building data done with {} patches loaded.".format(coords.shape[0]))

    # build model
    if args.load_model_from_path:
        model_path = os.path.join('./pretrained_models',args.arch)
    else:
        model_path = os.path.join('./random_w_models',args.arch)
        
    if os.path.isdir(model_path):
        model_file = os.listdir(model_path)
        model = torch.load(os.path.join(model_path,model_file[0]))
    
    model.backbone.conv1 = nn.Conv2d(image.shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.classifier[4] = nn.Conv2d(256, args.nb_class, kernel_size=(1, 1), stride=(1, 1))


    
    # model to gpu
    model = model.cuda()
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained_file_checkpoint):
        state_dict = torch.load(args.pretrained_file_checkpoint, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    cudnn.benchmark = True

    check_folder(os.path.join(args.model_dir,'prediction'))
    
    predict_network(test_loader, model, coords, pred_prob, pred_depth, 
                    stride, step_row, step_col, overlap, logger)

    logger.info("============ Inference finished ============")

def predict_network(dataloader, model, coords, pred_prob, pred_depth,
                     stride, step_row, step_col, overlap, logger):
      
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
            coord_x = coords[j:j+args.batch_size,0]
            coord_y = coords[j:j+args.batch_size,1]
            for b in range(c):
                pred_prob[coord_x[b]-st:coord_x[b]+st+stride%2,
                        coord_y[b]-st:coord_y[b]+st+stride%2] = out_batch[b,overlap//2:x-ovr+overlap%2,overlap//2:y-ovr+overlap%2]

                pred_depth[coord_x[b]-st:coord_x[b]+st+stride%2,
                        coord_y[b]-st:coord_y[b]+st+stride%2] = depth_out[b,0,overlap//2:x-ovr+overlap%2,overlap//2:y-ovr+overlap%2]

            
            j+=out_batch.shape[0] 
            
            
        raster_src = gdal.Open(args.ortho_image)
        
        row, col = raster_src.RasterYSize, raster_src.RasterXSize
        
        pred_prob = pred_prob[overlap//2:,overlap//2:]
        pred_prob = pred_prob[:row,:col]
        
        pred_depth = pred_depth[overlap//2:,overlap//2:]
        pred_depth = pred_depth[:row,:col]
        
        np.save(os.path.join(args.model_dir,'prediction','prob_map_itc{}_{}'.format(args.test_itc,args.overlap)), pred_prob)
        np.save(os.path.join(args.model_dir,'prediction','pred_class_itc{}_{}'.format(args.test_itc,args.overlap)), np.argmax(pred_prob,axis=-1))    
        np.save(os.path.join(args.model_dir,'prediction','depth_map_itc{}_{}'.format(args.test_itc,args.overlap)), pred_depth)

        
def define_loader(orto_img, lab = None):
    
    if not args.test_itc:
        image = read_tiff(orto_img)
        lab = np.ones(image.shape[1:])
        lab[np.sum(image,axis=0)==(11*image.shape[0])] = 0
        del image

    image = load_norm(orto_img)
        
    coords, _ = extract_patches_coord(lab, args.size_crops, args.overlap)
    image, stride, step_row, step_col, overlap, _, _ = add_padding_new(image, args.size_crops, args.overlap)

    return image, coords, stride, step_row, step_col, overlap


if __name__ == "__main__":
    overlaps = [0.1, 0.4, 0.6]
    for ov in overlaps:
        main(ov)
