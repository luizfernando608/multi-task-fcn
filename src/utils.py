# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import ast
import argparse
from logging import getLogger, CRITICAL
import pickle
import os

import numpy as np
import torch

# from .logger import create_logger, PD_Stats

import torch.distributed as dist

from osgeo import gdal, osr
import errno
import random
import matplotlib.pyplot as plt
import yaml

import gc

import matplotlib.pyplot as plt

from typing import Tuple

import warnings


plt.set_loglevel(level = 'critical')

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def array2raster(newRasterfn:str, dataset:gdal.Dataset, array:np.ndarray, dtype:str):
    """Save GTiff file from numpy.array

    Parameters
    ----------
    newRasterfn : string
        File path to save .tif
    dataset : 
        Dataset from gdal.Open()
    array : numpy.ndarray
        Array image to save as .tif
    dtype : string
        Data Type to save image
        Options: "Byte" or "Float32"
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()



def add_padding_new(img:np.ndarray, psize:int, overl:float, const:int = 0) -> Tuple:
    """Add padding to the image based on overlap and psize(patches size)

    Parameters
    ----------
    img : np.ndarray
        The image with n bands from remote sensing
        The fomat : (row, col, bands)
    psize : int
        The patch size to cut the segment image into boxes
    overl : float
        The overlap value that will have between the patches

    const : int, optional
        Contant to fill the paddin, by default 0

    Returns
    -------
    Tuple
    - pad_img : Image with padding
    - stride : The distance between each patch center
    - step_row
    - step_col
    - absolute_overlap = overl*psize
    - k1 : Num of patches in row axis
    - k2 : Num of patches in col axis
        
    """

    try:
        bands, row, col = img.shape

    except:
        bands = 0
        row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    # overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    # row += overlap//2
    # col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands > 0:
        npad_img = (
            (0,0), # padding to the band/channel axis
            (overlap//2, step_row+overlap), # padding to the row axis
            (overlap//2, step_col+overlap) # padding to the col axis
        )

    else:        
        npad_img = ((overlap//2, step_row+overlap), (overlap//2, step_col+overlap))  
        
    gc.collect()

    # padd with symetric (espelhado)
    pad_img = np.pad(img, npad_img, mode='constant', constant_values = const)

    gc.collect()
    
    # Number of patches: k1xk2
    k1, k2 = (row + step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap, k1, k2


def extract_patches_coord(img_gt:np.ndarray, 
                          psize:int, 
                          stride:int, 
                          step_row:int, 
                          step_col:int,
                          overl:float) -> np.ndarray:
    """
    The array of poisition of patches that will be used to evaluate the model

    Parameters
    ----------
    img_gt : np.ndarray
        ground truth segmentation
    psize : int
        The patch size to cut the segment image into boxes
    stride : int
        
    step_row : int
        
    step_col : int
        
    overl : float
        Overlap rate with the overlap that will have between patches
        
    Returns
    -------
    np.ndarray
        The coordinates of the center of each patch
    """
    
    # add padding to gt raster
    img_gt, stride, step_row, step_col, overlap,_, _ = add_padding_new(img_gt, psize, overl)
    
    overlap = int(round(psize * overl))
    
    row, col = img_gt.shape
    
    unique_class = np.unique(img_gt[img_gt!=0])
    
    if stride == 1:
    
        coords = np.where(img_gt!=0)
        coords = np.array(coords)
        coords = np.rollaxis(coords, 1, 0)

    else:
        # loop over x,y coordinates and extract patches
        coord_list = list()
    
        for m in range(psize//2, row - step_row - overlap, stride): 
            for n in range(psize//2, col - step_col - overlap, stride):
                
                coord = [m,n]
                
                class_patch = np.unique(img_gt[m - psize//2: m + psize//2, n - psize//2 : n+psize//2])
                
                if len(class_patch) > 1 or class_patch[0] in unique_class:
                    
                    coord_list.append(coord)                    


        coords = np.array(coord_list)
    
    return coords



def plot_figures(img_mult:np.ndarray, ref:np.ndarray, pred:np.ndarray, depth:np.ndarray, dist, model_dir:str, epoch:int, set_name:str):
    """Plot a comparison between the reference, prediction and depth images.

    Parameters
    ----------
    img_mult : np.ndarray
        Batch of images from remote sensing.
        The shape is (batch, bands, height, width).
    ref : np.ndarray
        Batch of references segmentation images.
        The shape is (batch, height, width).
    pred : np.ndarray
        Batch of predicted segmentation images.
        The shape is (batch, classes, height, width).
    depth : np.ndarray
        Batch of depth maps.
        The shape is (batch, height, width).
        This image has reference distance map used as reference.
    dist : _type_
        Batch of predicted depth maps.
        The shape is (batch, height, width).
        This image has predicted distance map generated by the model.
    model_dir : str
        model directory to save the images.
    epoch : int
        Current epoch to set in the image name.
    set_name : str
        First name to set in images name.
    """

    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
        dist = dist.data.cpu().numpy()

    if type(img_mult).__module__ != np.__name__:
        img_mult = img_mult.data.cpu().numpy()
        ref = ref.data.cpu().numpy()
        depth = depth.data.cpu().numpy()
    
    # Load the first 5 images in the batch
    batch = 5

    img_mult = img_mult[:batch,[5,3,2],:,:]
    img_mult = np.moveaxis(img_mult,1,3)

    ref = ref[:batch,:,:]
    pred_cl = np.argmax(pred[:batch,:,:,:],axis=1)+1
    pred_prob = np.amax(pred[:batch,:,:,:],axis=1)

    depth = depth[:batch,:,:]
    dist = dist[:batch,0,:,:]
    
    # pred_cl[ref==0] = 0

    nrows = 6
    ncols = batch
    imgs = [img_mult, ref, pred_cl, pred_prob, depth, dist]
    
    getLogger('matplotlib').setLevel(level=CRITICAL)
    warnings.filterwarnings("ignore")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(batch, nrows))
    
    cont = 0
    cont_img = 0

    for ax in axes.flat:
        ax.set_axis_off()

        if cont==0:
            # Set image from remote sensing
            ax.imshow(imgs[cont][cont_img], interpolation='nearest')

        elif cont==1 or cont==2:
            # Set image from reference and predicted segmentation
            ax.imshow(imgs[cont][cont_img], cmap='Dark2', interpolation='nearest', vmin=0, vmax=8)

        elif cont==3:
            # Set probability generated by the model
            ax.imshow(imgs[cont][cont_img], cmap='OrRd', interpolation='nearest',vmin=0, vmax=1)

        else:
            # Set the reference depth map and the predicted depth map
            ax.imshow(imgs[cont][cont_img], interpolation='nearest',vmin=0, vmax=1)
        
        cont_img+=1

        if cont_img == ncols:
            cont+=1
            cont_img=0

    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    axes[0,0].set_title("Real Image")
    axes[1,0].set_title("Ground Truth Segmentation")
    axes[2,0].set_title("Predicted Segmentation")
    axes[3,0].set_title("Probability Map")
    axes[4,0].set_title("Ground Truth Depth")
    axes[5,0].set_title("Predicted Depth")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()



def check_folder(folder_dir:str):
    """If the folder does not exist, create it.

    Parameters
    ----------
    folder_dir : str
        Folder directory.
    """
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    
    elif s.lower() in TRUTHY_STRINGS:
        return True
    
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


# def initialize_exp(params, *args, dump_params=True):
#     """
#     Initialize the experience:
#     - dump parameters
#     - create checkpoint repo
#     - create a logger
#     - create a panda object to keep track of the training statistics
#     """

#     # dump parameters
#     if dump_params:
#         pickle.dump(params, open(os.path.join(params.model_dir, "params.pkl"), "wb"))

#     # create a panda object to log loss and acc
#     training_stats = PD_Stats(
#         os.path.join(params.model_dir, "stats" + str(params.rank) + ".pkl"), args
#     )

#     # create a logger
#     logger = create_logger(
#         os.path.join(params.model_dir, "train.log"), rank=params.rank
#     )
#     logger.info("============ Initialized logger ============")
#     logger.info(
#         "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
#     )
#     logger.info("The experiment will be stored in %s\n" % params.model_dir)
#     logger.info("")
#     return logger, training_stats


def restart_from_checkpoint(ckp_paths:str, logger, run_variables:dict=None, **kwargs):
    """Load weights and hyperparameters from a checkpoint file in ckp_paths.
    If the checkpoint is not found, the model dont change run_variables and model state_dict

    Parameters
    ----------
    ckp_paths : str
        Path to the checkpoint file.
    logger : logging.Logger
        Logger to log the loading process.
    run_variables : dict, optional
        Hypertparameters to load from the checkpoint file, by default None

    """
    DEVICE = get_device()

    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        
        for ckp_path in ckp_paths:
            
            if os.path.isfile(ckp_path):
                break


    else:
        ckp_path = ckp_paths


    if not os.path.isfile(ckp_path):
        logger.info("No checkpoint found.")
        return


    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file and load to GPU
    checkpoint = torch.load(ckp_path, map_location=DEVICE)


    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():

        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)


            except TypeError:
                msg = value.load_state_dict(checkpoint[key])

            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))

            
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))


    # re load variable important for the run
    if run_variables is not None:

        for var_name in run_variables:

            if var_name in checkpoint:

                run_variables[var_name] = checkpoint[var_name]



def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Used to log loss and acc during training
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def read_tiff(tiff_file:str) -> np.ndarray:
    """Read tiff file and return a numpy array

    Parameters
    ----------
    tiff_file : str
        Path to the tiff file

    Returns
    -------
    np.ndarray
        Numpy array with the image

    Raises
    ------
    FileNotFoundError
        If the file is not found
    """
    
    # verify if file exist
    if not os.path.isfile(tiff_file):
        raise FileNotFoundError("File not found: {}".format(tiff_file))
    
    print(tiff_file)

    data = gdal.Open(tiff_file).ReadAsArray()

    return data



def load_norm(path, mask=[0], mask_indx = 0):
    """Read image from `path` divide all values by 255

    Parameters
    ----------
    path : str
        Path to load image
    mask : list, optional
        Deprecated, by default [0]
    mask_indx : int, optional
        Deprecated, by default 0

    Returns
    -------
    Image normalized
        Tensor image with the format [channels, row, cols]
    """
    image = read_tiff(path)

    if image.dtype != np.float32:
        image = np.float32(image)

    print("Image shape: ", image.shape, " Min value: ", image.min(), " Max value: ", image.max())
    if len(image.shape) < 3:
        image = np.expand_dims(image, 0)
    
    print("Before normalize, Min value: ", image.min(), " Max value: ", image.max())

    normalize(img = image)

    print("Normalize, Min value: ", image.min(), " Max value: ", image.max())

    return image

def filter_outliers(img, bins=10000, bth=0.01, uth=0.99, mask=[0], mask_indx=0):
    """
    Apply outlier filtering to image data using histogram-based thresholds.

    Parameters
    ----------
    img : numpy.ndarray
        The input image data with shape (bands, height, width).
    bins : int, optional
        The number of bins for histogram calculation. Default is 10000.
    bth : float, optional
        Lower threshold percentage for valid values. Default is 0.01.
    uth : float, optional
        Upper threshold percentage for valid values. Default is 0.99.
    mask : list or numpy.ndarray, optional
        Binary mask indicating regions of interest. Default is [0].
    mask_indx : int, optional
        Index of mask to use for outlier filtering. Default is 0.

    Returns
    -------
    numpy.ndarray
        Image data with outliers filtered within specified thresholds.

    Notes
    -----
    - NaN values are replaced with zeros before outlier filtering.
    - The function applies outlier filtering band-wise.
    - Use the mask parameter to specify regions for outlier filtering.
    - Outliers exceeding threshold values are clipped to the respective thresholds.
    """
    img[np.isnan(img)] = 0  # Filter NaN values.

    if len(mask) == 1:
        mask = np.zeros((img.shape[1:]), dtype='int64')

    for band in range(img.shape[0]):
        hist = np.histogram(img[:, :mask.shape[0], :mask.shape[1]][band, mask==mask_indx].ravel(), bins=bins)

        cum_hist = np.cumsum(hist[0]) / hist[0].sum()

        max_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < uth])]) / 100
        min_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < bth])]) / 100

        img[band, :,:,][img[band, :,:,] > max_value] = max_value
        img[band, :,:,][img[band, :,:,] < min_value] = min_value

    return img



def normalize(img:np.ndarray):
    """Normalize image inplace.
    Apply StandardScaler to image

    Parameters
    ----------
    img : np.ndarray
        Image array to normalize
        Shape: (BANDS, ROW, COL)
    """
    # iterate through channels and standardize
    for i in range(img.shape[0]):
        
        std = np.std(img[i], ddof=0)
        mean = np.mean(img[i])

        img[i] = (img[i]-mean)/std
    


def add_padding(img, psize, val = 0):
    '''
    DEPRECATED FUNCTION!
    Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        bands, row, col = img.shape
    
    except:
        bands = 0
        row, col = img.shape
    
    if bands>0:
        npad_img = ((0,0), (psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val

    else:        
        npad_img = ((psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val

    pad_img = np.pad(img, npad_img, mode='constant', constant_values=constant_values)

    return pad_img


def oversamp(coords:np.ndarray, lab:np.ndarray, under = False) -> np.ndarray:
    """Sample the data to balance the classes

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the segmentation samples, where the values are non-zero
    lab : np.ndarray
        The segmentation labels, where the values are non-zero
    under : bool, optional
        Define if the sampling is under or over.

        True: under sampling based on the median of the classes,
        False: over sampling based on the max number of samples in a class, 

        by default False

    Returns
    -------
    np.ndarray
        The coordinates of the segmentation samples, where the values are non-zero.
        Theses coordinates are balanced based on the tree-type classes.
    """

    
    uniq, count = np.unique(lab, return_counts=True)

    if under:
        max_samp = int(np.median(count))

    else:
        max_samp = np.max(count)

    
    out_coords = np.zeros( (max_samp*len(uniq), 2), dtype='int64')
    

    for j in range(len(uniq)):

        lab_ind = np.where(lab == uniq[j]) 

        # If num of samples where the class is present is less than max_samp
        # then we need to oversample
        if len(lab_ind[0]) < max_samp:
            # Randomly select samples with replacement to match max_samp
            index = np.random.choice(lab_ind[0], max_samp, replace=True)
            # Add to output array
            out_coords[j*max_samp:(j+1)*max_samp,:] = coords[index]
            
        # If the number of samples where the class is present is the same as max_samp
        # then we don't need to oversample, just add the samples randomly to the output array
        else:
            # Randomly select samples without replacement
            index = np.random.choice(lab_ind[0], max_samp, replace=False)
            # Add to output array
            out_coords[j*max_samp:(j+1)*max_samp,:] = coords[index]

            
    return out_coords


class AttrDict(dict):
    """Dictionary with attributes
    The dictionary values can be accessed as attributes

    Examples
    --------
    >>> d = AttrDict({'a':1, 'b':2})
    >>> d.a
    1
    """
    def __init__(self, *args, **kwargs):

        super(AttrDict, self).__init__(*args, **kwargs)

        self.__dict__ = self



def read_yaml(yaml_path:str)->dict:
    """Get the yaml file and convert to dict

    Parameters
    ----------
    yaml_path : str
        Path to the yaml file

    Returns
    -------
    dict
        Dictionary with keys and values from the yaml file
    """
    with open(yaml_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            yaml_attrdict = AttrDict()
            yaml_attrdict.update(yaml_dict)
        except yaml.YAMLError as exc:
            print(exc)
        
    # for each value try to convert to float
    for key in yaml_attrdict.keys():
        try:
            yaml_attrdict[key] = ast.literal_eval(yaml_attrdict[key])
        except:
            pass

    return yaml_attrdict



def print_sucess(message:str):
    """Print success message in green color
    
    Parameters:
    ----------
    message: str
        Message to print
    """
    print("\033[92m {}\033[00m" .format(message))