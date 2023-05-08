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


from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    read_tiff,
    load_norm,
    check_folder,
    plot_figures,
    oversamp
)

from src.multicropdataset import DatasetFromCoord
from src.resnet import ResUnet

logger = getLogger('swav_model')

parser = argparse.ArgumentParser(description="Training of MUlti-task FCN")

#########################
#### data parameters ####
#########################
parser.add_argument("--dump_path", type=str, default="./exp_deeplab_v4",
                    help="experiment dump path for checkpoints and log")
parser.add_argument('--image_path', type=str, default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/NNDiffusePanSharpening_REFLECTANCE_TIFF_NOCLOUDS_8bits.tif',
                        help="Path containing the raster image")
parser.add_argument('--ref_path',type=str, default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/ITC_CLASS.tif', 
                    help="Path containing the refrence label image for training")
parser.add_argument('--depth_path',type=str, default='D:/Projects/PUC-PoC/data_new/amazonas/ITC_annotation/train_depth.tif', 
                    help="Path containing the refrence depth image for training")
parser.add_argument("--samples", type=int, default=2500, 
                    help="nb samples per epoch")
parser.add_argument("--size_crops", type=int, default=256, 
                    help="Size of the input tile for the network")
parser.add_argument("--augment", type=bool, default=False, 
                    help="True for data augmentation during training")


#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="deeplabv3_resnet50", type=str, 
                    help="convnet architecture --> 'resunet','deeplabv3_resnet50'")
parser.add_argument("--pretrained", default=True, type=bool, 
                    help="True for load pretrained weights from Imagenet")
parser.add_argument("--frozen", default=False, type=bool, 
                    help="True for frozen the resnet backbone")
parser.add_argument("--filters", default=[32,32,32,32], type=int, 
                    help="Filter for the ResUnet for trained from scratch")


#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=30, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=16, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.01, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0.0001, help="final learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

##########################
#### others parameters ###
##########################

parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")
parser.add_argument("--seed", type=int, default=[31,10,75,102,40], help="seeds")


def main():
    global args, figures_path
    args = parser.parse_args()
    check_folder(args.dump_path)
    fix_random_seeds(args.seed[0])
    
    figures_path = os.path.join(args.dump_path, 'figures')
    check_folder(figures_path)

    
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    
    
    ######### Define Loader ############
    raster_train = read_tiff(args.ref_path)
    depth_img = read_tiff(args.depth_path)

    image, coords_train, raster_train, labs_coords_train = define_loader(args.moizaic_image, 
                                                            raster_train,
                                                            args.size_crops)    
    

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
    
    # build model
    if args.arch == 'resunet':    # trained ResUnet from scratch
        model = ResUnet(channel=image.shape[0], nb_classes = len(np.unique(raster_train)), 
                        filters=args.filters)


    # build model
    else:  
        # use deeplabv3_resnet50 pretrained or randomly initialized
        if args.pretrained:
            # create folder
            model_path = os.path.join('./pretrained_models',args.arch)
        else:
            model_path = os.path.join('./random_w_models',args.arch)
            
        if os.path.isdir(model_path):
            model_file = os.listdir(model_path)
            model = torch.load(os.path.join(model_path,model_file[0]))
        else:
            check_folder(model_path)
            model = torch.hub.load('pytorch/vision:v0.10.0', args.arch, 
                                   pretrained=args.pretrained,
                                   aux_loss =True)
            torch.save(model, os.path.join(model_path,'model'))
        
        # modify initial conv and classfiers
        model.backbone.conv1 = nn.Conv2d(image.shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        model.classifier[4] = nn.Conv2d(256, len(np.unique(labs_coords_train)), kernel_size=(1, 1), stride=(1, 1))


    # copy model to GPU
    model = model.cuda()
    
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd
    )
    
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")


    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]


    cudnn.benchmark = True

    best_val = 100.0
    cont_early = 0
    patience = 20

    for epoch in range(start_epoch, args.epochs):
        np.random.shuffle(train_loader.dataset.coord)

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train the network
        scores_tr = train(train_loader, model, optimizer, epoch, lr_schedule)
        
        training_stats.update(scores_tr)
        
        is_best = scores_tr[1] <= best_val
            
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if is_best:
                logger.info("============ Saving best models at epoch %i ... ============" % epoch)
                cont_early = 0
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                )
                best_val = scores_tr[1]
            else:
                cont_early+=1
                
                
            if cont_early == patience:
                logger.info("============ Early Stop at epoch %i ... ============" % epoch)
                break
            


def train(train_loader, model, optimizer, epoch, lr_schedule):

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
            plot_figures(inp_img,ref,soft(out_batch['out']),depth,
                         sig(out_batch['aux']),figures_path,epoch,'train')

            
    return (epoch, loss_avg.avg)


def define_loader(orto_img, gt_lab,  
                  size_crops,
                  test=False):
    if not test:
        image = load_norm(orto_img)
    
    # gt_lab must have zero for the background (unknown region)
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


if __name__ == "__main__":
    main()
