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
        
        if self.augment:
            # Define augmentation methods
            self.trans = K.AugmentationSequential(
                K.RandomResizedCrop(size=(self.psize,self.psize),scale=(0.5, 1.), p=0.0), #Randomly Crop image and resize to the current shape,
                # K.RandomRotation(degrees=(-45.0,45.0), p=0.2),
                # K.RandomElasticTransform(kernel_size=(3, 3), sigma=(5.0, 5.0), alpha=(1.0, 1.0),p=0.2),
                K.RandomPosterize(3., p=0.2), # Randomly reduce the number of bit of each color channe
                K.RandomEqualize(p=0.2), # Randomly change color histogram
                K.RandomMotionBlur(3, 10., 0.5), # Randomly apply blur to the image,
                # K.RandomSharpness(sharpness=0.5),
                K.RandomGaussianBlur((3, 3), (0.1, 2.0),p=0.5), # Randomly apply blur to the image,
                # K.RandomPerspective(0.1, p=0.5),
                # K.RandomThinPlateSpline(p=0.5),
            
                data_keys=['input','input','mask'], # define keyargs to the function
                
                keepdim = True, # keep the same image dim

                same_on_batch=False
                )


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
        
        # Crop image around the coords with a box with side psize
        image = self.image[:, 
                           self.coord[idx, 0] - self.psize//2 : self.coord[idx,0] + self.psize//2,
                           self.coord[idx,1] - self.psize//2 : self.coord[idx,1] + self.psize//2]
        

        image = torch.from_numpy(image.astype(np.float32))
        
        # If is training, return image, depth map, and label ref croppped
        if not self.evaluation:
            # crop the label ref with the same box
            ref = self.labels[self.coord[idx,0] - self.psize//2 : self.coord[idx,0] + self.psize//2,
                              self.coord[idx,1] - self.psize//2 : self.coord[idx,1] + self.psize//2]
            
            
            # crop the depth map with the same box
            depth = self.depth_img[self.coord[idx,0] - self.psize//2 : self.coord[idx,0] + self.psize//2,
                                   self.coord[idx,1] - self.psize//2 : self.coord[idx,1] + self.psize//2]


            ref = torch.tensor(ref.astype(np.float32))       
            
            depth = torch.tensor(depth.astype(np.float32)) 

            # If true, applies data augmentation
            if self.augment:
                
                image, depth, ref = self.trans(image, depth, ref)

            return image, depth, ref.long()

        # If evaluation is True, return just the image crop
        return image
