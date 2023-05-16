import os

import numpy as np
import torch
import torch.nn as nn

from .resnet import ResUnet
from .metrics import evaluate_metrics
from .utils import check_folder, load_norm, AverageMeter, plot_figures

from logging import Logger


def define_loader(orto_img, gt_lab, size_crops, test=False):
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
    The model can be pretrained or randomly initialized

    Parameters
    ----------
    image_shape : list
        List with the shape of the input image

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
            # create folder
            model_path = os.path.join('./pretrained_models', arch)
        else:
            model_path = os.path.join('./random_w_models', arch)
            
        if os.path.isdir(model_path):
            model_file = os.listdir(model_path)
            model = torch.load(os.path.join(model_path, model_file[0]))

        else:
            check_folder(model_path)
            model = torch.hub.load('pytorch/vision:v0.10.0', 
                arch, 
                pretrained=pretrained,
                aux_loss =True)
            
            torch.save(model, os.path.join(model_path,'model'))
        
        # modify initial conv and classfiers
        model.backbone.conv1 = nn.Conv2d(
            in_channels = image_shape[0],
            out_channels= 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        model.aux_classifier[4] = nn.Conv2d(
            in_channels = 256, 
            out_channels = 1, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )
        
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
    # load weights
    if os.path.isfile(checkpoint_file_path):

        state_dict = torch.load(checkpoint_file_path, map_location="cuda:0")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info(f'key "{k}" could not be found in provided state dict')
            
            elif state_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and provided state dict')
                state_dict[k] = v
        
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Load pretrained model with msg: {msg}")
    else:
        logger.info("No pretrained weights found => training with random weights")
    
    return model


def train(train_loader:torch.utils.data.DataLoader, model:nn.Module, optimizer:torch.optim.Optimizer, epoch:int, lr_schedule:np.array, figures_path:str, logger: Logger):
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
    model.train()
    loss_avg = AverageMeter()
    
    # define functions
    soft = nn.Softmax(dim=1).cuda()
    log_soft = nn.LogSoftmax(dim=1).cuda()    
    sig = nn.Sigmoid().cuda()
    
    # define losses
    criterion = nn.NLLLoss(reduction='none').cuda()
    aux_criterion = nn.L1Loss(reduction='none').cuda()

    for it, (inp_img, depth, ref) in enumerate(train_loader):      

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ forward pass and loss ... ============
        # compute model loss and output
        inp_img = inp_img.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        ref = ref.cuda(non_blocking=True)
        
        # creat mask for the unknown pixels
        mask = torch.ones(ref.shape)
        mask[ref==0] = 0
        mask = mask.cuda(non_blocking=True)

        ref_copy = torch.zeros(ref.shape).long().cuda(non_blocking=True)
        ref_copy[mask>0] = torch.sub(ref[mask>0], 1)
        
        # calculate losses
        out_batch = model(inp_img)
        loss1 = mask*criterion(log_soft(out_batch['out']), ref_copy)
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

        # Evaluate summaries only once in a while
        if it % 50 == 0:
            summary_batch=evaluate_metrics(soft(out_batch['out']), ref, -1)
            
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

            
    return (epoch, loss_avg.avg)



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
    best_acc : float
        Best accuracy achieved so far
    """
    save_dict = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "is_iter_finished": is_iter_finished,
        "best_acc": best_acc,
        "count_early": count_early,
    }
    torch.save(save_dict, last_checkpoint_path)