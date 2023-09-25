# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
import kornia.augmentation as K
import torch.nn as nn

from typing import Tuple

import os

import torchvision.transforms as transforms 

from tqdm import tqdm

logger = getLogger()


class DatasetFromCoord(Dataset):
    """Dataset to load training data"""
    def __init__(self, 
                image:np.ndarray = None,
                labels:np.ndarray = None,
                depth_img:np.ndarray = None,
                coords:np.ndarray = None, 
                psize: int = None,
                samples:int = False,
                evaluation:bool = False,
                augment:bool = False):
        """Initialize the dataset to load training data
        It defines the augmentation methods and the data to be loaded in the model
        This instance folows the pytorch Dataset pattern

        Parameters
        ----------
        image : np.ndarray, optional
            The real image from remote sensing, by default None
        labels : np.ndarray, optional
            The segmentation of trees in image, by default None
        depth_img : np.ndarray, optional
            The segmentation with filters like distance map and gaussian blur, by default None
        coords : np.ndarray, optional
            The positions in the ground_truth segmentation where the values are different of 0, by default None
        psize : int, optional
            The side of the square to be cut from the image.
            This box involves the position of the tree and the border of the box.
            The box is centered in the tree position.
            by default ``None``
        samples : int, optional
            Num samples per epoch, by default False
        evaluation : bool, optional
            Define the loading method for training or evaluation.
                True: return just the image crop,
                False: return image, depth map, and label ref croppped
            by default False
        augment : bool, optional
            If True applies data augmentation for training, by default False
        """

        super(DatasetFromCoord, self).__init__()
        self.image = image
        self.labels = labels
        self.depth_img = depth_img
        self.coord = coords
        self.psize = psize
        self.samples = samples
        self.evaluation = evaluation
        
        self.augment = augment
        

    def __len__(self):
        """Get the Dataset length

        Returns
        -------
        int
            The number of samples in the dataset
        """
        
        if self.samples:
            return self.samples
        
        
        else:
            return len(self.coord)
        
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The image crop, depth map crop, and label ref crop
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        if self.evaluation:
            # Crop image around the coords with a box with side psize
            image = self.image[:, 
                            self.coord[idx, 0] - self.psize//2 : self.coord[idx,0] + self.psize//2,
                            self.coord[idx,1] - self.psize//2 : self.coord[idx,1] + self.psize//2]
 

        
        # If is training, return image, depth map, and label ref croppped
        if not self.evaluation:

            # If augmentation is enabled, random crop the label around the position of the coord
            if self.augment:
                # random distance between the label coord and the left side bound box
                random_x_prop = np.random.uniform(low = 0.1, high = 0.9)
                # the distance between the label center and the left bound box
                dist_x_left_border = int(self.psize * random_x_prop)

                # random distance between the label coord and the top side bound box
                random_y_prop = np.random.uniform(low = 0.1, high = 0.9)
                # the distance between the label center and the right bound box
                dist_y_top_border = int(self.psize * random_y_prop)


            else:
                dist_x_left_border = self.psize//2

                dist_y_top_border = self.psize//2
            

            dist_x_right_border = self.psize - dist_x_left_border
            dist_y_bottom_border = self.psize - dist_y_top_border

            # Crop image around the coords with a box with side psize
            image = self.image[:, 
                            self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                            self.coord[idx, 1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]
            
            # crop the label ref with the same box
            ref = self.labels[self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                              self.coord[idx,1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]
            
            
            # crop the depth map with the same box
            depth = self.depth_img[self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                                   self.coord[idx,1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]

            

            image = torch.from_numpy(image.astype(np.float32))

            ref = torch.tensor(ref.astype(np.float32))       
            
            depth = torch.tensor(depth.astype(np.float32)) 


            # # If true, applies data augmentation
            if self.augment:
                # Run Horizontal Flip
                if np.random.random() > 0.5:
                    image = transforms.functional.hflip(image)
                    ref = transforms.functional.hflip(ref)
                    depth = transforms.functional.hflip(depth)

                # Run Vertical Flip
                if np.random.random() > 0.5:
                    image = transforms.functional.vflip(image)
                    ref = transforms.functional.vflip(ref)
                    depth = transforms.functional.vflip(depth)
                
                # Run random rotation
                angle = int(np.random.choice([0, 90, 180, 270]))
                
                image = transforms.functional.rotate(image.unsqueeze(0), angle).squeeze(0)
                ref = transforms.functional.rotate(ref.unsqueeze(0), angle).squeeze(0)
                depth = transforms.functional.rotate(depth.unsqueeze(0), angle).squeeze(0)


            return image, depth, ref.long()

        # If evaluation is True, return just the image crop
        return image


if __name__ == "__main__":

    import sys
    sys.path.append("/home/luiz/multi-task-fcn")

    from model import define_loader
    from utils import oversamp, read_yaml, read_tiff

    args = read_yaml("args.yaml")

    current_iter_folder = "/home/luiz/multi-task-fcn/1th_version_data/iter_001"
    current_iter = int(current_iter_folder.split("_")[-1])
    DATA_PATH = "/home/luiz/multi-task-fcn/1th_version_data"  

    ######### Define Loader ############
    LABEL_PATH = os.path.join(DATA_PATH, "segmentation","samples_A1_train2tif.tif" )
    raster_train = read_tiff(LABEL_PATH)

    DEPTH_PATH = os.path.join(DATA_PATH, "iter_000", "distance_map", "train_distance_map.tif")
    depth_img = read_tiff(DEPTH_PATH)

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
        augment = True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    np.random.shuffle(train_loader.dataset.coord)

    for batch in train_loader:
        batch
        pass