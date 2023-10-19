import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResUnet
from .metrics import evaluate_metrics, evaluate_f1
from .utils import check_folder, load_norm, AverageMeter, plot_figures, get_device

from typing import Tuple

import gc

from logging import Logger

from tqdm import tqdm

def define_loader(orto_img:str, gt_lab:np.ndarray, size_crops:int, test=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Define how the image will be loaded to the model

    Parameters
    ----------
    orto_img : str
        The file path to the map image. The image was generated from remote sensing
    gt_lab : np.ndarray
        The 2D array with segmentation labels
    size_crops : int
        The borders from the image to cut
    test : bool, optional
        Define if it is loaded for testing, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - image :  np.ndarray
            The image from remote sensing with shape [row, col, bands]
        - coords : np.ndarray
            The positions in the image where the values are different of 0.
        - gt_lab : 
            The segmentation image with borders removed
        - gt_lab[gt_lab!=0]
            The position where the segmentation is different of 0.
    """

    if not test:
        image = load_norm(orto_img)
    
    gt_lab[:size_crops,:] = 0
    gt_lab[-size_crops:,:] = 0
    gt_lab[:,:size_crops] = 0
    gt_lab[:,-size_crops:] = 0

    coords = np.where(gt_lab!=0)
    coords = np.array(coords)
    coords = np.rollaxis(coords, 1, 0)

    if test:
        return None, coords, gt_lab, None
    
    return image, coords, gt_lab, gt_lab[gt_lab!=0]


def build_model(image_shape:list, num_classes:int, arch:str, filters:list, pretrained:bool)->nn.Module:
    """Build model according to architecture
    The architecture can be 'resunet' or 'deeplabv3_resnet50'
    The model can be either pretrained or randomly initialized

    Parameters
    ----------
    image_shape : list
        List with the shape of the input image.
    num_classes : int
        Num of unique classes in the dataset
    arch : str
        Architecture name to build the model
        The architecture can be 'resunet' or 'deeplabv3_resnet50'
    filters : list
        List of number of filters for each block of the model, if the model is resunet
    pretrained : bool
        If True, the model is loaded from pretrained weights available in pytorch hub

    Returns
    -------
    nn.Module
        Pytorch model with the specified architecture
    """


    # build model
    if arch == 'resunet':    # trained ResUnet from scratch
        model = ResUnet(
            channel=image_shape[0], 
            nb_classes = num_classes, 
            filters=filters)

    # build model
    elif arch == "deeplabv3_resnet50":

        # use deeplabv3_resnet50 pretrained or randomly initialized
        if pretrained:
            # If is pretrained, the model is loaded from pretrained_models folder.
            # the weights model were downloaded from pytorch hub
            model_path = os.path.join('./pretrained_models', arch)

        else:
            # If is not pretrained, doesnt download/load model with pretrained weights
            model_path = os.path.join('./random_w_models', arch)


        if os.path.isdir(model_path):
            model_file = os.listdir(model_path)
            model = torch.load(os.path.join(model_path, model_file[0]))


        else:
            # If doesnt have the model, download from pytorch hub
            check_folder(model_path)
            model = torch.hub.load('pytorch/vision:v0.10.0', 
                arch, 
                pretrained = pretrained,
                aux_loss = True)
            
            torch.save(model, os.path.join(model_path,'model'))
        

        # modify initial conv and classfiers to the input image shape and num of classes
        model.backbone.conv1 = nn.Conv2d(
            in_channels = image_shape[0],
            out_channels= 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )

        # Adding an auxiliar classifier for distance map task
        model.aux_classifier[4] = nn.Conv2d(
            in_channels = 256, 
            out_channels = 1, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )
        
        # modify classifier to the num of classes
        model.classifier[4] = nn.Conv2d(
            in_channels = 256, 
            out_channels = num_classes, 
            kernel_size=(1, 1), 
            stride=(1, 1))
    
    else:
        raise ValueError(f"Unknown architecture {arch}.\nPlease choose among 'resunet' or 'deeplabv3_resnet50'")

    return model


def load_weights(model: nn.Module, checkpoint_file_path:str, logger: Logger)-> nn.Module:
    """Load weights for model from checkpoint file
    If the checkpoint file doesnt exist, the model is loaded with random weights

    Parameters
    ----------
    model : nn.Module
        Pytorch builded model
    checkpoint_file_path : str
        Path to checkpoint file
    logger : Logger
        Logger to log information

    Returns
    -------
    nn.Module
        Pytorch model with loaded weights
    """
    DEVICE = get_device()

    # load weights
    if os.path.isfile(checkpoint_file_path):

        state_dict = torch.load(checkpoint_file_path, map_location=DEVICE)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}


        # Execute some verifications about the stat dict loaded from checkpoint
        for k, v in model.state_dict().items():
            
            if k not in list(state_dict):
                logger.info(f'key "{k}" could not be found in provided state dict')
            
            elif state_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and provided state dict')
                state_dict[k] = v
        

        # Set the model weights
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Load pretrained model with msg: {msg}")

    
    else:
        logger.info("No pretrained weights found => training with random weights")
    
    return model



def categorical_focal_loss(input:torch.Tensor, target:torch.Tensor, gamma = 2) -> torch.Tensor:
    """Partial Categorical Focal Loss Implementation based on the paper 
    "Multi-task fully convolutional network for tree species
    mapping in dense forests using small training
    hyperspectral data"



    Parameters
    ----------
    input : torch.Tensor
        The output from ResNet Model without the classification layer 
        shape: (batch, class, image_height, image_width)

    target : torch.Tensor
        The ground_truth segmentation with index for each class
        shape: (batch, image_height, image_width)
    
    Returns
    -------
    torch.Tensor
        The loss for each pixel in image
        shape : (batch, image_height, image_width)
    """

    prob = F.softmax(input, dim = 1)
    log_prob = F.log_softmax(input, dim = 1)

    return F.nll_loss(
        ((1 - prob) ** gamma) * log_prob, 
        target=target,
        reduction = "none"
    )




def train(train_loader:torch.utils.data.DataLoader, 
          model:nn.Module, 
          optimizer:torch.optim.Optimizer, 
          epoch:int, 
          lr_schedule:np.ndarray, 
          figures_path:str, 
          logger: Logger):
    """Train model for one epoch

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Data loader for training
    model : nn.Module
        Pytorch model
    optimizer : torch.optim.Optimizer
        Pytorch optimizer
    current_epoch : int
        Current epoch to update current learning rate 
    lr_schedule : np.array
        Learning rate schedule to update at each iteration
    figures_path : str
        Path to save sample figures 

    Returns
    -------
    tuple[int, float]
        Tuple with epoch and average loss
    """
    DEVICE = get_device()

    model.train()
    loss_avg = AverageMeter()
    
    f1_avg = AverageMeter()

    # define functions
    soft = nn.Softmax(dim=1).to(DEVICE)
    sig = nn.Sigmoid().to(DEVICE)   

    # define losses
    # criterion = nn.NLLLoss(reduction='none').cuda()
    aux_criterion = nn.MSELoss(reduction='none').to(DEVICE)

    for it, (inp_img, depth, ref) in enumerate(tqdm(train_loader)):      

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ forward pass and loss ... ============
        # compute model loss and output
        inp_img = inp_img.to(DEVICE, non_blocking=True)
        depth = depth.to(DEVICE, non_blocking=True)
        ref = ref.to(DEVICE, non_blocking=True)
        
        # create mask for the unknown pixels
        mask = torch.where(ref == 0, torch.tensor(0.0), torch.tensor(1.0))
        mask = mask.to(DEVICE, non_blocking=True)

        ref_copy = torch.zeros(ref.shape).long().to(DEVICE, non_blocking=True)
        ref_copy[mask>0] = torch.sub(ref[mask>0], 1)
        
        # Foward Passs
        out_batch = model(inp_img)
        
        # loss1 = mask*categorical_focal_loss_2(out_batch["out"], ref_copy, alpha = 1)

        loss1 = mask*categorical_focal_loss(out_batch["out"], ref_copy)

        loss2 = mask*aux_criterion(sig(out_batch['aux'])[:,0,:,:], depth)
        
        loss = (loss1 + loss2)/2 
        loss = torch.sum(loss)/ref[ref>0].shape[0]

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        
        # performs updates using calculated gradients
        optimizer.step()
        
        # update the average loss
        loss_avg.update(loss.item())

        f1_avg.update( evaluate_f1(soft(out_batch['out']), ref) )

        gc.collect()

        # Evaluate summaries only once in a while
        if it % 50 == 0:
            summary_batch = evaluate_metrics(soft(out_batch['out']), ref)
            
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    loss=loss_avg,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            logger.info(summary_batch)
            
        if it == 0:
            # plot samples results for visual inspection
            plot_figures(inp_img, ref, soft(out_batch['out']),depth,
                         sig(out_batch['aux']),figures_path,epoch,'train')

            
    return (epoch, f1_avg.avg)



def save_checkpoint(last_checkpoint_path:str, model:nn.Module, optimizer:torch.optim.Optimizer, epoch:int, best_acc:float, count_early:int, is_iter_finished=False):
    """Save model checkpoint at last_checkpoint_path

    Parameters
    ----------
    last_checkpoint_path : str
        File path to save checkpoint
    model : nn.Module
        Pytorch model at the end of epoch
    optimizer : torch.optim.Optimizer
        Pytorch optimizer at the end of epoch
    epoch : int
        Current epoch
    best_val : float
        Best accuracy achieved so far
    """
    save_dict = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "is_iter_finished": is_iter_finished,
        "best_val": best_acc,
        "count_early": count_early,
    }
    torch.save(save_dict, last_checkpoint_path)