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

logger = getLogger()


class DatasetFromCoord(Dataset):
    def __init__(self, 
                image = None,
                labels = None,
                depth_img = None,
                coords = None, 
                psize = None,
                samples=False,
                evaluation = False,
                augment = False):

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
            self.trans = K.AugmentationSequential(K.RandomResizedCrop(size=(self.psize,self.psize),scale=(0.5, 1.), p=0.0),
                                              # K.RandomRotation(degrees=(-45.0,45.0), p=0.2),
                                              # K.RandomElasticTransform(kernel_size=(3, 3), sigma=(5.0, 5.0), alpha=(1.0, 1.0),p=0.2),
                                              K.RandomPosterize(3., p=0.2),
                                              K.RandomEqualize(p=0.2),
                                              K.RandomMotionBlur(3, 10., 0.5),
                                              # K.RandomSharpness(sharpness=0.5),
                                              K.RandomGaussianBlur((3, 3), (0.1, 2.0),p=0.5),
                                              # K.RandomPerspective(0.1, p=0.5),
                                              # K.RandomThinPlateSpline(p=0.5),
                                              data_keys=['input','input','mask'],
                                              keepdim = True,
                                              same_on_batch=False)
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.coord)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.image[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2]
        image = torch.from_numpy(image.astype(np.float32))
        
        if not self.evaluation:
            ref = self.labels[self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2,
                                   self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2]
            
            depth = self.depth_img[self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2,
                                   self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2]

        
            ref = torch.tensor(ref.astype(np.float32))       
            depth = torch.tensor(depth.astype(np.float32)) 

            if self.augment:
                image, depth, ref = self.trans(image,depth,ref)

            return image, depth, ref.long()
        
        return image
